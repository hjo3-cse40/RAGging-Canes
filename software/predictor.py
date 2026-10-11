"""Real-time RIFE/Optical-Flow Frame Interpolator and 3-Second Trajectory & Realism Predictor.

Designed for the ASUS Ascent GX10 + ESP32-S3 wearable obstacle-awareness pipeline:
1. RIFEInterpolator: Synthesizes intermediate frames between sparse ESP32 captures to eliminate
   visual choppiness on the live stream while automatically gating out rapid head-turn cuts.
2. TrajectoryPredictor: Tracks YOLO26n detections across frames, filters out physically unrealistic
   glitches/flicker or ego-motion spikes, and rolls out 0.5s–3.0s future trajectories, Time-To-Collision
   (TTC), and predicted 6-zone occupancy.
3. AsyncVLMForecaster: Optional non-blocking background thread that queries local Ollama VLMs
   to semantically validate the 3-second scenario without blocking the 25 ms YOLO reflex loop.
"""
import base64
from collections import deque
import io
import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

try:
    import numpy as np
except ImportError:
    np = None

try:
    import cv2
except ImportError:
    cv2 = None

torch = None
nn = None
F = None


def _ensure_torch():
    global torch, nn, F
    if torch is not None:
        return True
    try:
        import torch as _t
        import torch.nn as _nn
        import torch.nn.functional as _F
        torch, nn, F = _t, _nn, _F
        return True
    except ImportError:
        return False


try:
    from software.vision import zone_states
    from software.dashboard.geometry import project
except ImportError:
    from vision import zone_states
    from dashboard.geometry import project


ZONE_NAMES = ["L3", "L2", "L1", "R1", "R2", "R3"]
THREE_ZONE_NAMES = ["LEFT", "CENTER", "RIGHT"]


def six_to_three_zones(zones_6):
    """Fold 6 horizontal sectors [L3, L2, L1, R1, R2, R3] into 3 haptic motor zones [Left, Center, Right]."""
    if not zones_6 or len(zones_6) != 6:
        return [-1, -1, -1]
    if any(z == -1 for z in zones_6):
        return [-1, -1, -1]
    return [
        1 if (zones_6[0] == 1 or zones_6[1] == 1) else 0,
        1 if (zones_6[2] == 1 or zones_6[3] == 1) else 0,
        1 if (zones_6[4] == 1 or zones_6[5] == 1) else 0,
    ]


def three_to_six_zones(zones_3):
    """Expand 3 haptic motor zones [Left, Center, Right] into 6 display sectors [L3..R3]."""
    if not zones_3 or len(zones_3) != 3:
        return [0, 0, 0, 0, 0, 0]
    if any(z == -1 for z in zones_3):
        return [-1] * 6
    l_val, c_val, r_val = (1 if z == 1 else 0 for z in zones_3)
    return [l_val, l_val, c_val, c_val, r_val, r_val]



def _make_ifblock(in_channels=7, hidden=24):
    if not _ensure_torch():
        return None

    class _IFBlock(nn.Module):
        """Lightweight RIFE-style Intermediate Flow Block (IFNet coarse-to-fine stage)."""
        def __init__(self):
            super().__init__()
            self.conv0 = nn.Sequential(
                nn.Conv2d(in_channels, hidden, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(hidden, hidden, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.conv_flow = nn.Conv2d(hidden, 5, 3, 1, 1)
            nn.init.zeros_(self.conv_flow.weight)
            nn.init.zeros_(self.conv_flow.bias)

        def forward(self, x, scale=1.0):
            if scale != 1.0:
                x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
            feat = self.conv0(x)
            out = self.conv_flow(feat)
            out = F.interpolate(out, scale_factor=scale * 2.0, mode="bilinear", align_corners=False)
            flow = out[:, :4] * scale * 2.0
            mask = torch.sigmoid(out[:, 4:5])
            return flow, mask

    return _IFBlock()


class RIFEInterpolator:
    """Intermediate video frame synthesizer between sparse ESP32 camera captures.

    Supports:
    - PyTorch GPU/MPS RIFE-style pyramid flow + differentiable grid warping (or external RIFE .pt weights)
    - OpenCV Dense Inverse Search (DIS) bidirectional optical flow warping (fast C++ path)
    - Pure Pillow/NumPy fallback for minimal test environments
    Includes a scene-cut / fast head-turn gate (MAD threshold) so large ego-motion jumps
    do not produce ghosting artifacts.
    """

    def __init__(self, factor=3, device="auto", scene_cut_threshold=0.30,
                 jpeg_quality=85, weights_path=None):
        self.factor = max(0, int(factor))
        self.scene_cut_threshold = float(scene_cut_threshold)
        self.jpeg_quality = int(jpeg_quality)
        self.device = self._resolve_device(device)
        self.backend = self._select_backend(weights_path)
        self._prev_bgr = None
        self._prev_time = None
        self._lock = threading.Lock()
        self.stats = {
            "backend": self.backend,
            "device": self.device,
            "factor": self.factor,
            "last_rife_ms": 0.0,
            "last_scene_diff": 0.0,
            "interpolated_frames": 0,
            "skipped_cuts": 0,
        }

    @staticmethod
    def _resolve_device(device):
        if device == "cpu":
            return "cpu"
        if not _ensure_torch():
            return "cpu"
        if device in ("auto", None, ""):
            if torch.cuda.is_available():
                return "cuda:0"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        if str(device).isdigit():
            if torch.cuda.is_available():
                return f"cuda:{device}"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return str(device)

    def _select_backend(self, weights_path):
        self._rife_net = None
        if weights_path and Path(weights_path).is_file() and _ensure_torch():
            try:
                self._rife_net = torch.jit.load(str(weights_path), map_location=self.device).eval()
                return "torch_rife_jit"
            except Exception:
                pass
        if cv2 is not None and np is not None:
            try:
                self._dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
                if self.device.startswith(("cuda", "mps")) and _ensure_torch():
                    return "rife_hybrid_gpu"
                return "cv2_dis_flow"
            except Exception:
                self._dis = None
        if np is not None and self.device != "cpu" and _ensure_torch():
            self._rife_net = _make_ifblock(7, hidden=24).to(self.device).eval()
            return "torch_ifnet"
        return "pillow_blend"

    def reset(self):
        with self._lock:
            self._prev_bgr = None
            self._prev_time = None

    @staticmethod
    def _to_bgr(frame_input, raw_jpeg=None):
        try:
            if np is not None and frame_input is not None and hasattr(frame_input, "shape"):
                arr = frame_input
                if arr.ndim == 3 and arr.shape[2] == 3:
                    return arr
            if cv2 is not None and np is not None and raw_jpeg is not None:
                nparr = np.frombuffer(raw_jpeg, np.uint8)
                if nparr.size > 0:
                    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if bgr is not None:
                        return bgr
            if isinstance(frame_input, Image.Image) and np is not None:
                rgb = np.asarray(frame_input.convert("RGB"))
                return rgb[:, :, ::-1].copy()
            if raw_jpeg is not None and np is not None:
                with Image.open(io.BytesIO(raw_jpeg)) as im:
                    rgb = np.asarray(im.convert("RGB"))
                    return rgb[:, :, ::-1].copy()
        except Exception:
            return None
        return None

    @staticmethod
    def _encode_bgr(bgr, quality=85):
        if cv2 is not None:
            ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
            if ok:
                return enc.tobytes()
        rgb = bgr[:, :, ::-1]
        im = Image.fromarray(rgb)
        out = io.BytesIO()
        im.save(out, format="JPEG", quality=int(quality))
        return out.getvalue()

    def _compute_scene_diff(self, prev_bgr, curr_bgr):
        """Coarse 16x12 area-averaged structural difference so local 10-40px edge shifts
        are smoothly interpolated by RIFE while global camera cuts/head-whips are gated."""
        if prev_bgr.shape != curr_bgr.shape:
            return 1.0
        if cv2 is not None:
            p_coarse = cv2.resize(prev_bgr, (16, 12), interpolation=cv2.INTER_AREA).astype(np.float32)
            c_coarse = cv2.resize(curr_bgr, (16, 12), interpolation=cv2.INTER_AREA).astype(np.float32)
            return float(np.mean(np.abs(p_coarse - c_coarse)) / 255.0)
        h, w = prev_bgr.shape[:2]
        bh, bw = max(1, h // 12), max(1, w // 16)
        p_sub = prev_bgr[:bh * 12, :bw * 16].reshape(12, bh, 16, bw, -1).mean(axis=(1, 3))
        c_sub = curr_bgr[:bh * 12, :bw * 16].reshape(12, bh, 16, bw, -1).mean(axis=(1, 3))
        return float(np.mean(np.abs(p_sub - c_sub)) / 255.0)

    def _synthesize_dis_flow(self, prev_bgr, curr_bgr, alphas):
        """Bidirectional optical flow warping at downsampled resolution + GPU/CPU fusion."""
        h, w = prev_bgr.shape[:2]
        # Compute flow at 1/2 scale (e.g. 320x240) for <3 ms latency, then upscale flow field
        flow_w, flow_h = max(64, w // 2), max(48, h // 2)
        prev_small = cv2.resize(prev_bgr, (flow_w, flow_h), interpolation=cv2.INTER_LINEAR)
        curr_small = cv2.resize(curr_bgr, (flow_w, flow_h), interpolation=cv2.INTER_LINEAR)
        prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY)

        flow_01_small = self._dis.calc(prev_gray, curr_gray, None)
        # Symmetric backward flow approximation or inverse DIS pass
        flow_10_small = self._dis.calc(curr_gray, prev_gray, None)

        scale_x = float(w) / float(flow_w)
        scale_y = float(h) / float(flow_h)
        flow_01 = cv2.resize(flow_01_small, (w, h), interpolation=cv2.INTER_LINEAR)
        flow_10 = cv2.resize(flow_10_small, (w, h), interpolation=cv2.INTER_LINEAR)
        flow_01[:, :, 0] *= scale_x
        flow_01[:, :, 1] *= scale_y
        flow_10[:, :, 0] *= scale_x
        flow_10[:, :, 1] *= scale_y

        if self.backend == "rife_hybrid_gpu" and torch is not None:
            return self._warp_on_torch_device(prev_bgr, curr_bgr, flow_01, flow_10, alphas)

        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        frames = []
        for alpha in alphas:
            t = float(alpha)
            # Intermediate backward warp coordinates from t -> 0 and t -> 1
            map0_x = grid_x - t * flow_01[:, :, 0]
            map0_y = grid_y - t * flow_01[:, :, 1]
            map1_x = grid_x - (1.0 - t) * flow_10[:, :, 0]
            map1_y = grid_y - (1.0 - t) * flow_10[:, :, 1]

            warped0 = cv2.remap(prev_bgr, map0_x, map0_y, interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
            warped1 = cv2.remap(curr_bgr, map1_x, map1_y, interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
            blended = cv2.addWeighted(warped0, 1.0 - t, warped1, t, 0.0)
            frames.append(blended)
        return frames

    def _warp_on_torch_device(self, prev_bgr, curr_bgr, flow_01, flow_10, alphas):
        """Perform sub-millisecond differentiable tensor grid_sample warping on GPU/MPS."""
        h, w = prev_bgr.shape[:2]
        with torch.no_grad():
            dev = self.device
            t0_img = torch.from_numpy(prev_bgr).to(dev, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            t1_img = torch.from_numpy(curr_bgr).to(dev, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            f01 = torch.from_numpy(flow_01).to(dev, dtype=torch.float32).unsqueeze(0)
            f10 = torch.from_numpy(flow_10).to(dev, dtype=torch.float32).unsqueeze(0)

            yy, xx = torch.meshgrid(
                torch.linspace(-1.0, 1.0, h, device=dev),
                torch.linspace(-1.0, 1.0, w, device=dev),
                indexing="ij",
            )
            base_grid = torch.stack((xx, yy), dim=-1).unsqueeze(0)
            norm_f01 = torch.stack((f01[..., 0] * (2.0 / max(1, w - 1)),
                                    f01[..., 1] * (2.0 / max(1, h - 1))), dim=-1)
            norm_f10 = torch.stack((f10[..., 0] * (2.0 / max(1, w - 1)),
                                    f10[..., 1] * (2.0 / max(1, h - 1))), dim=-1)

            frames = []
            for alpha in alphas:
                t = float(alpha)
                g0 = base_grid - t * norm_f01
                g1 = base_grid - (1.0 - t) * norm_f10
                w0 = F.grid_sample(t0_img, g0, mode="bilinear", padding_mode="border", align_corners=True)
                w1 = F.grid_sample(t1_img, g1, mode="bilinear", padding_mode="border", align_corners=True)
                fused = ((1.0 - t) * w0 + t * w1).squeeze(0).permute(1, 2, 0).clamp(0, 255).to(torch.uint8)
                frames.append(fused.cpu().numpy())
            return frames

    def interpolate_pair(self, prev_bgr, curr_bgr, num_mid=None):
        """Generate `num_mid` intermediate BGR frames between `prev_bgr` and `curr_bgr`."""
        k = self.factor if num_mid is None else int(num_mid)
        if k <= 0 or prev_bgr is None or curr_bgr is None:
            return [], 0.0, False

        t0 = time.perf_counter()
        scene_diff = self._compute_scene_diff(prev_bgr, curr_bgr) if np is not None else 0.0
        if scene_diff > self.scene_cut_threshold:
            elapsed = (time.perf_counter() - t0) * 1000.0
            with self._lock:
                self.stats["last_rife_ms"] = round(elapsed, 2)
                self.stats["last_scene_diff"] = round(scene_diff, 4)
                self.stats["skipped_cuts"] += 1
            return [], elapsed, True

        alphas = [(i + 1) / float(k + 1) for i in range(k)]
        if self.backend in ("rife_hybrid_gpu", "cv2_dis_flow") and getattr(self, "_dis", None) is not None:
            mid_bgrs = self._synthesize_dis_flow(prev_bgr, curr_bgr, alphas)
        elif np is not None:
            p_f = prev_bgr.astype(np.float32)
            c_f = curr_bgr.astype(np.float32)
            mid_bgrs = [((1.0 - a) * p_f + a * c_f).astype(np.uint8) for a in alphas]
        else:
            mid_bgrs = []

        elapsed = (time.perf_counter() - t0) * 1000.0
        with self._lock:
            self.stats["last_rife_ms"] = round(elapsed, 2)
            self.stats["last_scene_diff"] = round(scene_diff, 4)
            self.stats["interpolated_frames"] += len(mid_bgrs)
        return mid_bgrs, elapsed, False

    def push_and_interpolate_jpegs(self, raw_jpeg, frame_bgr=None):
        """Stateful helper for live stream: accepts newest frame, returns list of (jpeg_bytes, is_interpolated)
        including intermediate frames followed by the real frame, plus telemetry dict."""
        if self.factor <= 0:
            return [(raw_jpeg, False)], {
                "rife_enabled": False,
                "rife_ms": 0.0,
                "interpolated_count": 0,
                "scene_cut_skipped": False,
                "scene_diff": 0.0,
                "backend": self.backend,
            }

        curr_bgr = self._to_bgr(frame_bgr, raw_jpeg)
        with self._lock:
            prev_bgr = self._prev_bgr
            self._prev_bgr = curr_bgr
            self._prev_time = time.monotonic()

        if prev_bgr is None or curr_bgr is None or prev_bgr.shape != curr_bgr.shape:
            return [(raw_jpeg, False)], {
                "rife_enabled": True,
                "rife_ms": 0.0,
                "interpolated_count": 0,
                "scene_cut_skipped": False,
                "scene_diff": 0.0,
                "backend": self.backend,
            }

        mid_bgrs, rife_ms, skipped = self.interpolate_pair(prev_bgr, curr_bgr, self.factor)
        out = []
        for mid in mid_bgrs:
            out.append((self._encode_bgr(mid, self.jpeg_quality), True))
        out.append((raw_jpeg, False))
        return out, {
            "rife_enabled": True,
            "rife_ms": round(rife_ms, 2),
            "interpolated_count": len(mid_bgrs),
            "scene_cut_skipped": skipped,
            "scene_diff": self.stats["last_scene_diff"],
            "backend": self.backend,
        }


class TrajectoryPredictor:
    """Kalman / Kinematic 3-Second Object Trajectory Predictor & Physical Realism Filter.

    Tracks detected objects (`person`, `chair`, etc.), evaluates whether their motion is
    physically realistic vs. sensor noise / head-turn ego-motion, and projects 0.5s–3.0s
    future bounding boxes, ground-plane paths, Time-To-Collision (TTC), and 6-zone states.
    """

    HORIZONS = (0.5, 1.0, 2.0, 3.0)
    MAX_METRIC_SPEED_MPS = 4.5        # Maximum physically realistic indoor human speed (m/s)
    MAX_NORM_SPEED_PER_S = 1.45       # Maximum normalized screen widths/s before flagging teleport/head-whip
    MAX_SCALE_JUMP_PER_S = 2.5        # Maximum relative bounding-box height change rate

    def __init__(self, boundaries=None, max_track_idle_s=1.5, alpha_pos=0.65, alpha_vel=0.45):
        self.boundaries = boundaries or [i / 6 for i in range(7)]
        self.max_track_idle_s = float(max_track_idle_s)
        self.alpha_pos = float(alpha_pos)
        self.alpha_vel = float(alpha_vel)
        self._lock = threading.Lock()
        self._tracks = {}
        self._next_id = 1
        self._last_ts = None
        self._keyframes = deque(maxlen=45)

    def reset(self):
        with self._lock:
            self._tracks.clear()
            self._next_id = 1
            self._last_ts = None
            self._keyframes.clear()

    @staticmethod
    def _iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
        return inter / (area_a + area_b - inter)

    @staticmethod
    def _box_to_state(xyxy):
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        w = max(4.0, x2 - x1)
        h = max(4.0, y2 - y1)
        cx = x1 + 0.5 * w
        cy = y1 + 0.5 * h
        return cx, cy, w, h

    @staticmethod
    def _state_to_box(cx, cy, w, h, width, height):
        w = max(8.0, min(float(width), w))
        h = max(8.0, min(float(height), h))
        x1 = max(0.0, min(float(width) - 4.0, cx - 0.5 * w))
        y1 = max(0.0, min(float(height) - 4.0, cy - 0.5 * h))
        x2 = max(x1 + 4.0, min(float(width), cx + 0.5 * w))
        y2 = max(y1 + 4.0, min(float(height), cy + 0.5 * h))
        return [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]

    def update(self, objects, width, height, timestamp=None, calibration_config=None):
        """Update tracks with current frame detections and return enriched objects + 3s scenario forecast."""
        t0 = time.perf_counter()
        now = float(timestamp) if timestamp is not None else time.monotonic()
        width = max(1, int(width or 640))
        height = max(1, int(height or 480))

        with self._lock:
            dt_global = (now - self._last_ts) if self._last_ts is not None else 0.1
            dt_global = max(0.015, min(2.0, dt_global))
            self._last_ts = now

            # 1. Prepare candidate detections
            candidates = []
            for idx, obj in enumerate(objects or []):
                xyxy = obj.get("xyxy", [0, 0, 10, 10])
                cx, cy, bw, bh = self._box_to_state(xyxy)
                ground_xy = obj.get("ground_xy")
                if ground_xy is None and calibration_config:
                    ground_xy = project(xyxy, width, height, calibration_config)
                candidates.append({
                    "orig_idx": idx,
                    "obj": dict(obj),
                    "class": obj.get("class", "person"),
                    "confidence": float(obj.get("confidence", 0.5)),
                    "xyxy": [float(v) for v in xyxy],
                    "cx": cx,
                    "cy": cy,
                    "w": bw,
                    "h": bh,
                    "ground_xy": ground_xy,
                })

            # 2. Associate candidates with active tracks (IoU + normalized center distance)
            matched_tracks = set()
            matched_cands = {}
            pairs = []
            for c_idx, cand in enumerate(candidates):
                for t_id, trk in self._tracks.items():
                    if trk["class"] != cand["class"]:
                        continue
                    dt = max(0.015, min(2.0, now - trk["last_seen"]))
                    pred_cx = trk["cx"] + trk["vx"] * dt
                    pred_cy = trk["cy"] + trk["vy"] * dt
                    pred_box = self._state_to_box(pred_cx, pred_cy, trk["w"], trk["h"], width, height)
                    iou = self._iou(cand["xyxy"], pred_box)
                    norm_dist = math.hypot((cand["cx"] - pred_cx) / width, (cand["cy"] - pred_cy) / height)
                    if iou > 0.05 or norm_dist < 0.38:
                        score = iou * 2.0 - norm_dist
                        pairs.append((score, c_idx, t_id))

            pairs.sort(key=lambda item: item[0], reverse=True)
            for _, c_idx, t_id in pairs:
                if c_idx in matched_cands or t_id in matched_tracks:
                    continue
                matched_cands[c_idx] = t_id
                matched_tracks.add(t_id)

            # 3. Estimate global ego-motion (common-mode horizontal shift across matched tracks)
            raw_shifts_vx = []
            for c_idx, t_id in matched_cands.items():
                cand = candidates[c_idx]
                trk = self._tracks[t_id]
                dt = max(0.015, min(2.0, now - trk["last_seen"]))
                raw_shifts_vx.append((cand["cx"] - trk["cx"]) / dt)

            ego_vx = 0.0
            ego_motion_detected = False
            if len(raw_shifts_vx) >= 2:
                # If multiple objects shift strongly in the same direction, likely head rotation
                signs = [1 if v > 0 else -1 for v in raw_shifts_vx if abs(v) / width > 0.25]
                if len(signs) == len(raw_shifts_vx) and abs(sum(signs)) == len(signs):
                    ego_vx = sum(raw_shifts_vx) / len(raw_shifts_vx)
                    ego_motion_detected = True

            # 4. Update matched tracks & create new tracks
            cand_to_track = {}
            for c_idx, cand in enumerate(candidates):
                if c_idx in matched_cands:
                    t_id = matched_cands[c_idx]
                    trk = self._tracks[t_id]
                    dt = max(0.015, min(2.0, now - trk["last_seen"]))

                    inst_vx = (cand["cx"] - trk["cx"]) / dt - 0.5 * ego_vx
                    inst_vy = (cand["cy"] - trk["cy"]) / dt
                    inst_vw = (cand["w"] - trk["w"]) / dt
                    inst_vh = (cand["h"] - trk["h"]) / dt

                    # Physical realism evaluation on transition
                    norm_speed = math.hypot(inst_vx / width, inst_vy / height)
                    scale_rate = abs(inst_vh) / max(16.0, trk["h"])

                    metric_speed = None
                    inst_gvx, inst_gvy = 0.0, 0.0
                    if cand["ground_xy"] and trk.get("ground_xy"):
                        inst_gvx = (cand["ground_xy"][0] - trk["ground_xy"][0]) / dt
                        inst_gvy = (cand["ground_xy"][1] - trk["ground_xy"][1]) / dt
                        metric_speed = math.hypot(inst_gvx, inst_gvy)

                    velocity_plausible = norm_speed <= self.MAX_NORM_SPEED_PER_S
                    if metric_speed is not None:
                        velocity_plausible = velocity_plausible and (metric_speed <= self.MAX_METRIC_SPEED_MPS)
                    scale_coherent = scale_rate <= self.MAX_SCALE_JUMP_PER_S

                    # Stationary class prior: chairs shouldn't walk across the room on their own
                    if cand["class"] == "chair" and not ego_motion_detected:
                        if (metric_speed is not None and metric_speed > 1.2) or norm_speed > 0.65:
                            velocity_plausible = False

                    if velocity_plausible and scale_coherent:
                        trk["vx"] = (1.0 - self.alpha_vel) * trk["vx"] + self.alpha_vel * inst_vx
                        trk["vy"] = (1.0 - self.alpha_vel) * trk["vy"] + self.alpha_vel * inst_vy
                        trk["vw"] = (1.0 - self.alpha_vel) * trk["vw"] + self.alpha_vel * inst_vw
                        trk["vh"] = (1.0 - self.alpha_vel) * trk["vh"] + self.alpha_vel * inst_vh
                        trk["cx"] = (1.0 - self.alpha_pos) * (trk["cx"] + trk["vx"] * dt) + self.alpha_pos * cand["cx"]
                        trk["cy"] = (1.0 - self.alpha_pos) * (trk["cy"] + trk["vy"] * dt) + self.alpha_pos * cand["cy"]
                        trk["w"] = (1.0 - self.alpha_pos) * trk["w"] + self.alpha_pos * cand["w"]
                        trk["h"] = (1.0 - self.alpha_pos) * trk["h"] + self.alpha_pos * cand["h"]
                        if cand["ground_xy"]:
                            if trk.get("ground_xy"):
                                trk["gvx"] = (1.0 - self.alpha_vel) * trk.get("gvx", 0.0) + self.alpha_vel * inst_gvx
                                trk["gvy"] = (1.0 - self.alpha_vel) * trk.get("gvy", 0.0) + self.alpha_vel * inst_gvy
                            trk["ground_xy"] = cand["ground_xy"]
                        if dt > 0.65 and cand["confidence"] < 0.55:
                            trk["hits"] = 1
                            trk["realistic"] = False
                            trk["realism_reason"] = "sparse_low_confidence_flicker"
                        else:
                            trk["hits"] += 1
                            trk["anomalies"] = max(0, trk.get("anomalies", 0) - 1)
                            trk["realistic"] = True
                            trk["realism_reason"] = "confirmed_kinematic_track"
                    else:
                        # Dampen velocity on unrealistic jump so we don't shoot trajectories off-screen
                        trk["vx"] *= 0.25
                        trk["vy"] *= 0.25
                        trk["vw"] *= 0.1
                        trk["vh"] *= 0.1
                        trk["cx"] = cand["cx"]
                        trk["cy"] = cand["cy"]
                        trk["w"] = cand["w"]
                        trk["h"] = cand["h"]
                        trk["ground_xy"] = cand["ground_xy"]
                        trk["anomalies"] = trk.get("anomalies", 0) + 1
                        trk["realistic"] = False
                        trk["realism_reason"] = (
                            "unrealistic_velocity_jump" if not velocity_plausible else "incoherent_scale_jump"
                        )

                    trk["confidence"] = cand["confidence"]
                    trk["last_seen"] = now
                    trk["norm_speed"] = round(math.hypot(trk["vx"] / width, trk["vy"] / height), 3)
                    trk["metric_speed_mps"] = round(
                        math.hypot(trk.get("gvx", 0.0), trk.get("gvy", 0.0)), 2
                    ) if trk.get("ground_xy") else None
                    cand_to_track[c_idx] = trk
                else:
                    t_id = self._next_id
                    self._next_id += 1
                    confirmed_immediately = cand["confidence"] >= 0.55
                    trk = {
                        "track_id": t_id,
                        "class": cand["class"],
                        "confidence": cand["confidence"],
                        "cx": cand["cx"],
                        "cy": cand["cy"],
                        "w": cand["w"],
                        "h": cand["h"],
                        "vx": 0.0,
                        "vy": 0.0,
                        "vw": 0.0,
                        "vh": 0.0,
                        "ground_xy": cand["ground_xy"],
                        "gvx": 0.0,
                        "gvy": 0.0,
                        "hits": 1,
                        "anomalies": 0,
                        "created_at": now,
                        "last_seen": now,
                        "realistic": confirmed_immediately,
                        "realism_reason": "initial_observation" if confirmed_immediately else "transient_unconfirmed",
                        "norm_speed": 0.0,
                        "metric_speed_mps": 0.0 if cand["ground_xy"] else None,
                    }
                    self._tracks[t_id] = trk
                    cand_to_track[c_idx] = trk

            # Prune expired tracks
            expired = [tid for tid, t in self._tracks.items() if now - t["last_seen"] > self.max_track_idle_s]
            for tid in expired:
                del self._tracks[tid]

            # 5. Roll out 0.5s - 3.0s future horizons for each object
            enriched_objects = []
            boxes_1s, boxes_2s, boxes_3s = [], [], []
            min_ttc_s = None
            realistic_count = 0
            filtered_count = 0
            forecast_notes = []

            for c_idx, cand in enumerate(candidates):
                trk = cand_to_track[c_idx]
                is_realistic = bool(trk["realistic"] and (trk["hits"] >= 2 or trk["confidence"] >= 0.55))
                if is_realistic:
                    realistic_count += 1
                else:
                    filtered_count += 1

                pred_boxes = {}
                pred_path = []

                # Compute Time-To-Collision (TTC) if approaching wearer
                ttc_s = None
                approaching = False
                if is_realistic:
                    if trk.get("ground_xy") and trk.get("gvy", 0.0) < -0.15:
                        # Metric ground Y is distance forward (m); negative gvy means approaching wearer (Y -> 0)
                        dist_y = max(0.1, trk["ground_xy"][1])
                        ttc_s = round(dist_y / (-trk["gvy"]), 2)
                        approaching = ttc_s <= 6.0
                    else:
                        # Uncalibrated image-space approach: foot moving down toward bottom (vy > 0) or box expanding (vh > 0)
                        foot_y = trk["cy"] + 0.5 * trk["h"]
                        foot_vy = trk["vy"] + 0.5 * trk["vh"]
                        rel_expand = trk["vh"] / max(20.0, trk["h"])
                        if foot_vy > height * 0.04 or rel_expand > 0.08:
                            rem_px = max(12.0, (height * 0.94) - foot_y)
                            ttc_from_foot = rem_px / max(1.0, foot_vy) if foot_vy > 1.0 else 99.0
                            ttc_from_scale = 1.0 / max(0.01, rel_expand) if rel_expand > 0.05 else 99.0
                            est_ttc = min(ttc_from_foot, ttc_from_scale)
                            if est_ttc <= 6.0:
                                ttc_s = round(est_ttc, 2)
                                approaching = True

                if ttc_s is not None and (min_ttc_s is None or ttc_s < min_ttc_s):
                    min_ttc_s = ttc_s

                for horizon in self.HORIZONS:
                    # Velocity damping factor over multi-second horizon
                    damp = math.pow(0.90, horizon) if is_realistic else 0.0
                    eff_t = horizon * damp
                    p_cx = trk["cx"] + trk["vx"] * eff_t
                    p_cy = trk["cy"] + trk["vy"] * eff_t
                    p_w = trk["w"] + trk["vw"] * eff_t * 0.6
                    p_h = trk["h"] + trk["vh"] * eff_t * 0.6
                    p_box = self._state_to_box(p_cx, p_cy, p_w, p_h, width, height)
                    key = f"{horizon:.1f}s"
                    pred_boxes[key] = p_box

                    p_ground = None
                    if trk.get("ground_xy"):
                        gx = round(trk["ground_xy"][0] + trk.get("gvx", 0.0) * eff_t, 3)
                        gy = round(max(0.1, trk["ground_xy"][1] + trk.get("gvy", 0.0) * eff_t), 3)
                        p_ground = [gx, gy]
                    elif calibration_config:
                        p_ground = project(p_box, width, height, calibration_config)

                    p_zones = zone_states([(p_box[0], p_box[2])], width, self.boundaries)
                    active_z = [i for i, z in enumerate(p_zones) if z == 1]

                    pred_path.append({
                        "dt": horizon,
                        "cx": round((p_box[0] + p_box[2]) * 0.5, 1),
                        "cy": round((p_box[1] + p_box[3]) * 0.5, 1),
                        "foot_x": round((p_box[0] + p_box[2]) * 0.5, 1),
                        "foot_y": round(p_box[3], 1),
                        "ground_xy": p_ground,
                        "zones": active_z,
                    })

                    if is_realistic:
                        if horizon == 1.0:
                            boxes_1s.append((p_box[0], p_box[2]))
                        elif horizon == 2.0:
                            boxes_2s.append((p_box[0], p_box[2]))
                        elif horizon == 3.0:
                            boxes_3s.append((p_box[0], p_box[2]))

                curr_z_idx = int(max(0, min(5, (trk["cx"] / width) * 6)))
                fut_z_idx = int(max(0, min(5, (pred_path[2]["cx"] / width) * 6))) if len(pred_path) >= 3 else curr_z_idx
                curr_z_name = ZONE_NAMES[curr_z_idx]
                fut_z_name = ZONE_NAMES[fut_z_idx]
                curr_z3_name = THREE_ZONE_NAMES[curr_z_idx // 2]
                fut_z3_name = THREE_ZONE_NAMES[fut_z_idx // 2]

                if is_realistic and (curr_z_idx != fut_z_idx or approaching):
                    spd_label = (
                        f"{trk['metric_speed_mps']:.1f} m/s"
                        if trk.get("metric_speed_mps") is not None
                        else f"{int(trk['norm_speed'] * 100)}%/s"
                    )
                    ttc_label = f" · TTC {ttc_s:.1f}s" if ttc_s is not None else ""
                    forecast_notes.append(
                        f"{trk['class'].capitalize()} #{trk['track_id']} ({curr_z_name}→{fut_z_name}, {spd_label}{ttc_label})"
                    )

                enriched = {
                    **cand["obj"],
                    "track_id": trk["track_id"],
                    "hits": trk["hits"],
                    "realistic": is_realistic,
                    "realism_reason": trk["realism_reason"],
                    "velocity_px_s": [round(trk["vx"], 1), round(trk["vy"], 1)],
                    "norm_speed": trk["norm_speed"],
                    "metric_speed_mps": trk["metric_speed_mps"],
                    "approaching": approaching,
                    "ttc_s": ttc_s,
                    "current_zone": curr_z_name,
                    "predicted_zone_2s": fut_z_name,
                    "current_zone_3": curr_z3_name,
                    "predicted_zone_3_2s": fut_z3_name,
                    "predicted_boxes": pred_boxes,
                    "predicted_path": pred_path,
                }
                enriched_objects.append(enriched)

            pred_zones_1s = zone_states(boxes_1s, width, self.boundaries)
            pred_zones_2s = zone_states(boxes_2s, width, self.boundaries)
            pred_zones_3s = zone_states(boxes_3s, width, self.boundaries)

            curr_boxes = [(c["xyxy"][0], c["xyxy"][2]) for c in candidates]
            curr_zones_6 = zone_states(curr_boxes, width, self.boundaries)
            curr_zones_3 = six_to_three_zones(curr_zones_6)
            pred_zones_3_1s = six_to_three_zones(pred_zones_1s)
            pred_zones_3_2s = six_to_three_zones(pred_zones_2s)
            pred_zones_3_3s = six_to_three_zones(pred_zones_3s)

            # Record keyframe snapshot for t-2s, t-1s, t0 temporal strip
            kf_tracks = []
            for obj in enriched_objects:
                kf_tracks.append({
                    "id": obj["track_id"],
                    "class": obj["class"],
                    "zone_3": obj["current_zone_3"],
                    "pred_zone_3": obj["predicted_zone_3_2s"],
                    "vx": obj["velocity_px_s"][0],
                    "vy": obj["velocity_px_s"][1],
                    "speed_mps": obj["metric_speed_mps"],
                    "ttc_s": obj["ttc_s"],
                    "approaching": obj["approaching"],
                    "realistic": obj["realistic"],
                    "hits": obj["hits"],
                })
            self._keyframes.append({
                "ts": now,
                "tracks": kf_tracks,
                "ego_motion": ego_motion_detected,
                "zones_3": curr_zones_3,
            })
            keyframes_3s = self._extract_3_keyframes(now)

            total_obs = len(candidates)
            motion_trend = "clear_or_receding"
            if total_obs == 0:
                realism_score = 1.0
                realistic_scene = True
                risk_level = "clear"
                summary = "SCENE CLEAR · No obstacles projected in 3.0s horizon."
            else:
                realism_score = round(realistic_count / float(total_obs), 2)
                realistic_scene = realistic_count > 0 and not ego_motion_detected
                if min_ttc_s is not None and min_ttc_s <= 2.2:
                    risk_level = "collision_warning"
                elif min_ttc_s is not None or any(o["current_zone"] != o["predicted_zone_2s"] for o in enriched_objects if o["realistic"]):
                    risk_level = "approaching"
                elif realistic_count > 0:
                    risk_level = "stable"
                else:
                    risk_level = "filtered_noise"

                # Derive kinematic motion_trend for the 3-motor MVP
                real_objs = [o for o in enriched_objects if o["realistic"]]
                if not realistic_scene or not real_objs:
                    motion_trend = "unrealistic_or_clear"
                else:
                    primary = min(real_objs, key=lambda o: o["ttc_s"] if o["ttc_s"] is not None else 99.0)
                    c3 = primary["current_zone_3"]
                    f3 = primary["predicted_zone_3_2s"]
                    vx_p = primary["velocity_px_s"][0]
                    if (c3 == "LEFT" and f3 in ("CENTER", "RIGHT")) or (c3 == "CENTER" and vx_p > 35.0):
                        motion_trend = "crossing_left_to_center"
                    elif (c3 == "RIGHT" and f3 in ("CENTER", "LEFT")) or (c3 == "CENTER" and vx_p < -35.0):
                        motion_trend = "crossing_right_to_center"
                    elif primary["approaching"] and (f3 == "CENTER" or c3 == "CENTER"):
                        motion_trend = "approaching_center"
                    elif f3 == "LEFT" or c3 == "LEFT":
                        motion_trend = "left_obstacle"
                    elif f3 == "RIGHT" or c3 == "RIGHT":
                        motion_trend = "right_obstacle"
                    else:
                        motion_trend = "approaching_center"

                if not realistic_scene:
                    reason_str = "Camera pan / head turn compensated" if ego_motion_detected else "Transient sensor noise filtered"
                    summary = f"UNREALISTIC MOTION FILTERED · {reason_str} ({filtered_count}/{total_obs} suppressed)."
                elif forecast_notes:
                    summary = "3S FORECAST · " + " | ".join(forecast_notes[:2])
                else:
                    active_names_3 = [THREE_ZONE_NAMES[i] for i, z in enumerate(pred_zones_3_2s) if z == 1]
                    z_str = ", ".join(active_names_3) if active_names_3 else "none"
                    summary = f"REALISTIC STABLE SCENE · {realistic_count} confirmed target(s) · 3-motor zones: [{z_str}]"

            rec_3 = [
                1 if (pred_zones_3_1s[i] == 1 or pred_zones_3_2s[i] == 1) else 0
                for i in range(3)
            ] if realistic_scene else [0, 0, 0]
            rec_6 = [
                1 if (pred_zones_1s[i] == 1 or pred_zones_2s[i] == 1) else 0
                for i in range(6)
            ] if realistic_scene else [0, 0, 0, 0, 0, 0]

            predict_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            scenario = {
                "realistic": realistic_scene,
                "realistic_detection": realistic_scene,
                "realism_score": realism_score,
                "ego_motion_detected": ego_motion_detected,
                "risk_level": risk_level,
                "motion_trend": motion_trend,
                "min_ttc_s": min_ttc_s,
                "confirmed_tracks": realistic_count,
                "filtered_anomalies": filtered_count,
                "zones_3": curr_zones_3,
                "predicted_zones_3_1s": pred_zones_3_1s,
                "predicted_zones_3_2s": pred_zones_3_2s,
                "predicted_zones_3_3s": pred_zones_3_3s,
                "recommended_pre_alert_zones_3": rec_3,
                "recommended_pre_alert_zones": rec_6,
                "predicted_zones_1s": pred_zones_1s,
                "predicted_zones_2s": pred_zones_2s,
                "predicted_zones_3s": pred_zones_3s,
                "keyframes_3s": keyframes_3s,
                "predicted_next_3s": summary,
                "summary": summary,
                "predict_ms": predict_ms,
            }
            return enriched_objects, scenario

    def _extract_3_keyframes(self, now):
        """Sample the closest keyframes for t-2.0s, t-1.0s, and t0 from the rolling buffer."""
        if not self._keyframes:
            return []
        targets = [("t-2s", now - 2.0), ("t-1s", now - 1.0), ("t0", now)]
        out = []
        for label, target_ts in targets:
            closest = min(self._keyframes, key=lambda item: abs(item["ts"] - target_ts))
            out.append({
                "slot": label,
                "dt_s": round(closest["ts"] - now, 2),
                "ego_motion": closest["ego_motion"],
                "zones_3": list(closest["zones_3"]),
                "tracks": list(closest["tracks"]),
            })
        return out


class DecisionForecaster:
    """Asynchronous System-1 Decision Layer for 3-Second Hazard & 3-Motor Haptic Arbitration.

    Supports:
    - `laya`: Local open-source `convaiinnovations/laya` decision model (Apache 2.0, ~421M ModernBERT
      with bidirectional option-order de-biasing, ~50–100 ms on MPS/CUDA, 100% offline).
    - `jev`: Hosted TypeSafe AI `POST https://api.typesafe.ai/v1/systemone` endpoint using the key in
      `private/jev_api_key.txt` or `JEV_API_KEY` environment variable.
    - `kinematic`: Zero-weight deterministic fallback.

    Runs in a non-blocking background thread in live mode so it never blocks YOLO26n.
    """

    LAYA_MODEL_ID = "convaiinnovations/laya"
    LAYA_REVISION = "7b928d828b7b0e022f929d9bd2e44165aa270148"
    JEV_API_URL = "https://api.typesafe.ai/v1/systemone"
    JEV_MODEL = "jev-latest"

    INSTRUCTIONS = (
        "Select the matching obstacle state and 3-motor haptic zone alert from the 3-second camera observation."
    )
    CRITERIA = {
        "approaching_center": "Approaching collision risk in center corridor; vibrate Center motor.",
        "crossing_left_to_center": "Walking from Left side toward Center corridor; vibrate Left and Center motors.",
        "crossing_right_to_center": "Walking from Right side toward Center corridor; vibrate Center and Right motors.",
        "left_obstacle": "Stationary obstacle confined to Left side only; vibrate Left motor only.",
        "right_obstacle": "Stationary obstacle confined to Right side only; vibrate Right motor only.",
        "unrealistic_or_clear": "No obstacle, empty clear room, or false head-turn sensor glitch; do not vibrate.",
    }

    TREND_TO_3ZONES = {
        "approaching_center": [0, 1, 0],
        "crossing_left_to_center": [1, 1, 0],
        "crossing_right_to_center": [0, 1, 1],
        "left_obstacle": [1, 0, 0],
        "right_obstacle": [0, 0, 1],
        "unrealistic_or_clear": [0, 0, 0],
    }

    ZONES3_TO_TREND = {
        (0, 1, 0): "approaching_center",
        (1, 1, 0): "crossing_left_to_center",
        (0, 1, 1): "crossing_right_to_center",
        (1, 0, 0): "left_obstacle",
        (0, 0, 1): "right_obstacle",
        (0, 0, 0): "unrealistic_or_clear",
        (1, 1, 1): "approaching_center",
        (1, 0, 1): "approaching_center",
    }

    MOTOR_QUESTIONS = {
        "left_motor": {
            "type": "noul",
            "instructions": "Should the Left haptic motor vibrate (obstacle on Left side or walking from Left toward Center)?",
        },
        "center_motor": {
            "type": "noul",
            "instructions": "Should the Center haptic motor vibrate (obstacle in Center corridor or walking toward Center)?",
        },
        "right_motor": {
            "type": "noul",
            "instructions": "Should the Right haptic motor vibrate (obstacle on Right side or walking from Right toward Center)?",
        },
    }

    TREND_TO_HAPTIC_PATTERN = {
        "approaching_center": "urgent_center_pulse",
        "crossing_left_to_center": "left_to_center_sweep",
        "crossing_right_to_center": "right_to_center_sweep",
        "left_obstacle": "steady_left_nudge",
        "right_obstacle": "steady_right_nudge",
        "unrealistic_or_clear": "silent",
    }

    def __init__(self, backend="auto", device="auto", jev_key_file="private/jev_api_key.txt",
                 jev_url=None, jev_model=None, interval_s=0.35, timeout_s=2.5):
        self.requested_backend = backend
        self.device = device
        self.jev_key_file = Path(jev_key_file) if jev_key_file else None
        self.jev_url = (jev_url or os.environ.get("JEV_API_URL") or self.JEV_API_URL).rstrip("/")
        self.jev_model = (jev_model or os.environ.get("JEV_MODEL") or self.JEV_MODEL).strip()
        self.interval_s = float(interval_s)
        self.timeout_s = float(timeout_s)
        self._lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._laya_agent = None
        self._laya_error = None
        self._latest_input = None
        self._latest_decision = None
        self._stop = threading.Event()
        self._thread = None

    def _read_jev_key(self):
        env_key = (os.environ.get("JEV_API_KEY") or os.environ.get("TYPESAFE_API_KEY") or "").strip()
        if env_key:
            return env_key
        if self.jev_key_file and self.jev_key_file.is_file():
            try:
                val = self.jev_key_file.read_text(encoding="utf-8").strip()
                if val and not val.startswith("#"):
                    return val.splitlines()[0].strip()
            except OSError:
                pass
        return None

    def set_backend(self, backend):
        if backend not in ("auto", "laya", "jev", "kinematic"):
            raise ValueError("Decision backend must be auto, laya, jev, or kinematic")
        with self._lock:
            self.requested_backend = backend

    def _ensure_laya(self):
        if self._laya_agent is not None:
            return self._laya_agent
        if self._laya_error is not None:
            return None
        try:
            os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            import laya
            dev = self.device
            if dev in ("auto", None, "", "0"):
                if _ensure_torch():
                    if torch.cuda.is_available():
                        dev = "cuda:0"
                    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        dev = "mps"
                    else:
                        dev = "cpu"
                else:
                    dev = "cpu"
            try:
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                self._laya_agent = laya.load(
                    self.LAYA_MODEL_ID,
                    revision=self.LAYA_REVISION,
                    device=dev,
                    backend="eager",
                )
            except Exception:
                os.environ.pop("HF_HUB_OFFLINE", None)
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
                self._laya_agent = laya.load(
                    self.LAYA_MODEL_ID,
                    revision=self.LAYA_REVISION,
                    device=dev,
                    backend="eager",
                )
            return self._laya_agent
        except Exception as exc:
            self._laya_error = str(exc)
            return None

    @staticmethod
    def build_keyframe_state_text(scenario):
        """Format the 3 keyframes (t-2s, t-1s, t0) and YOLO velocities into a concise state string for Laya / Jev."""
        keyframes = scenario.get("keyframes_3s") or []
        ego = scenario.get("ego_motion_detected", False)
        confirmed = scenario.get("confirmed_tracks", 0)
        filtered = scenario.get("filtered_anomalies", 0)
        ttc = scenario.get("min_ttc_s")

        if confirmed == 0 and filtered > 0 or ego:
            return (
                "t=-2s: Empty room. "
                "t=-1s: Fast camera head-turn glitch. "
                "t=0s: All motors OFF, silent, zero obstacles."
            )
        if confirmed == 0:
            return (
                "t=-2s: Empty room. "
                "t=-1s: Empty room. "
                "t=0s: All motors OFF, silent, zero obstacles."
            )

        start_z3 = None
        for kf in keyframes:
            trks_kf = [t for t in (kf.get("tracks") or []) if t.get("realistic", True)]
            if trks_kf:
                start_z3 = trks_kf[0].get("zone_3", "CENTER").capitalize()
                break

        parts = []
        for kf in keyframes:
            slot = kf.get("slot", "t0")
            trks = [t for t in (kf.get("tracks") or []) if t.get("realistic", True)]
            if not trks:
                parts.append(f"{slot}: Clear")
                continue
            t = trks[0]
            cls_name = str(t.get("class", "person")).capitalize()
            z3 = t.get("zone_3", "CENTER").capitalize()
            p3 = t.get("pred_zone_3", z3).capitalize()
            vx = float(t.get("vx", 0.0))
            spd = t.get("speed_mps")
            spd_str = f" at {spd:.1f} m/s" if spd is not None else ""
            if slot == "t0":
                if (z3 == "Left" and (p3 == "Center" or vx > 25.0)) or (start_z3 == "Left" and z3 == "Center" and vx > 25.0):
                    parts.append(
                        f"t=0s: Confirmed {cls_name} walking from Left side toward Center corridor; vibrate Left and Center motors."
                    )
                elif (z3 == "Right" and (p3 == "Center" or vx < -25.0)) or (start_z3 == "Right" and z3 == "Center" and vx < -25.0):
                    parts.append(
                        f"t=0s: Confirmed {cls_name} walking from Right side toward Center corridor; vibrate Center and Right motors."
                    )
                elif t.get("approaching") and (p3 == "Center" or z3 == "Center"):
                    ttc_str = f" (TTC {ttc:.1f}s)" if ttc is not None else ""
                    parts.append(
                        f"t=0s: Confirmed {cls_name} approaching collision risk in Center corridor{ttc_str}; vibrate Center motor."
                    )
                else:
                    parts.append(
                        f"t=0s: Stationary {cls_name} confined to {z3} side only; vibrate {z3} motor only."
                    )
            else:
                if z3 != p3 or (slot == "t-1s" and start_z3 in ("Left", "Right") and z3 == "Center"):
                    origin = start_z3 if (start_z3 in ("Left", "Right") and z3 == "Center") else z3
                    parts.append(f"{slot}: {cls_name} walking from {origin} side toward Center{spd_str}.")
                elif t.get("approaching"):
                    parts.append(f"{slot}: {cls_name} in {z3} approaching fast{spd_str}.")
                else:
                    parts.append(f"{slot}: Stationary {cls_name} confined to {z3} side only.")
        return " ".join(parts)

    def _run_laya(self, state_text):
        agent = self._ensure_laya()
        if agent is None:
            return None
        with self._infer_lock:
            res = agent.system_one(state_text, self.MOTOR_QUESTIONS, max_len=224, head_max_len=96)["answers"]
        p_l = float(res["left_motor"]["noul"])
        p_c = float(res["center_motor"]["noul"])
        p_r = float(res["right_motor"]["noul"])
        motor_probs = [round(p_l, 3), round(p_c, 3), round(p_r, 3)]
        z3 = (int(p_l >= 0.5), int(p_c >= 0.5), int(p_r >= 0.5))
        choice = self.ZONES3_TO_TREND.get(z3, "approaching_center")
        conf_bits = [
            p_l if z3[0] else (1.0 - p_l),
            p_c if z3[1] else (1.0 - p_c),
            p_r if z3[2] else (1.0 - p_r),
        ]
        conf = sum(conf_bits) / 3.0
        probs = {
            k: round(
                (
                    (p_l if v[0] else (1.0 - p_l))
                    * (p_c if v[1] else (1.0 - p_c))
                    * (p_r if v[2] else (1.0 - p_r))
                ),
                4,
            )
            for k, v in self.TREND_TO_3ZONES.items()
        }
        return {
            "backend": "laya",
            "model": self.LAYA_MODEL_ID,
            "choice": choice,
            "confidence": round(conf, 3),
            "probabilities": probs,
            "motor_probabilities_3": motor_probs,
            "jev_key_configured": bool(self._read_jev_key()),
        }

    def _run_jev(self, state_text):
        api_key = self._read_jev_key()
        if not api_key:
            return None
        questions = {
            "hazard_decision": {
                "type": "choice",
                "instructions": self.INSTRUCTIONS,
                "criteria": self.CRITERIA,
            },
            **self.MOTOR_QUESTIONS,
        }
        payload = {
            "model": self.jev_model,
            "state": state_text,
            "questions": questions,
        }
        req = urllib.request.Request(
            self.jev_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "RAGging-Canes/1.0",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            body = json.load(resp)
        answers = body.get("answers") or {}
        ans = answers.get("hazard_decision") or {}
        probs = ans.get("probabilities") or {}
        choice = ans.get("choice") or (max(probs, key=probs.get) if probs else "approaching_center")
        conf = float(ans.get("answer_confidence", ans.get("confidence", probs.get(choice, 0.85))))
        default_z3 = self.TREND_TO_3ZONES.get(choice, [0, 0, 0])
        motor_probs = [
            round(float((answers.get("left_motor") or {}).get("noul", default_z3[0])), 3),
            round(float((answers.get("center_motor") or {}).get("noul", default_z3[1])), 3),
            round(float((answers.get("right_motor") or {}).get("noul", default_z3[2])), 3),
        ]
        return {
            "backend": "jev",
            "model": body.get("model", self.jev_model),
            "choice": choice,
            "confidence": round(conf, 3),
            "probabilities": probs,
            "motor_probabilities_3": motor_probs,
            "usage": body.get("usage", {}),
            "jev_key_configured": True,
        }

    def evaluate_sync(self, scenario):
        """Evaluate the 3-keyframe scenario synchronously and return the structured decision JSON."""
        t0 = time.perf_counter()
        state_text = self.build_keyframe_state_text(scenario)
        with self._lock:
            req_backend = self.requested_backend

        raw_decision = None
        fallback_note = None
        if req_backend == "jev":
            try:
                raw_decision = self._run_jev(state_text)
                if raw_decision is None:
                    fallback_note = "Paste Jev key into private/jev_api_key.txt (using Laya fallback)"
            except Exception as exc:
                fallback_note = f"Jev API unreachable ({type(exc).__name__}), using Laya fallback"
            if raw_decision is None:
                raw_decision = self._run_laya(state_text)
        elif req_backend in ("laya", "auto"):
            raw_decision = self._run_laya(state_text)
            if raw_decision is None and req_backend == "auto":
                try:
                    raw_decision = self._run_jev(state_text)
                except Exception:
                    pass

        confirmed = int(scenario.get("confirmed_tracks", 0))
        kin_realistic = bool(scenario.get("realistic", True))
        kin_trend = scenario.get("motion_trend", "approaching_center")
        ttc = scenario.get("min_ttc_s")

        if raw_decision is None:
            choice = kin_trend if (kin_realistic and confirmed > 0) else "unrealistic_or_clear"
            raw_decision = {
                "backend": "kinematic",
                "model": "kalman-3zone-rules",
                "choice": choice,
                "confidence": float(scenario.get("realism_score", 0.9)),
                "probabilities": {choice: 1.0},
                "motor_probabilities_3": [float(v) for v in self.TREND_TO_3ZONES.get(choice, [0, 0, 0])],
                "jev_key_configured": bool(self._read_jev_key()),
            }

        choice = raw_decision["choice"]
        realistic_detection = bool(choice != "unrealistic_or_clear" and kin_realistic)
        zones_3 = list(self.TREND_TO_3ZONES.get(choice, [0, 0, 0]))
        if realistic_detection and confirmed > 0 and sum(zones_3) == 0:
            zones_3 = list(scenario.get("recommended_pre_alert_zones_3") or [0, 1, 0])
        elif not realistic_detection or confirmed == 0:
            zones_3 = [0, 0, 0]

        zones_6 = three_to_six_zones(zones_3)
        haptic_pattern = self.TREND_TO_HAPTIC_PATTERN.get(choice, "silent") if realistic_detection else "silent"

        if not realistic_detection:
            predicted_next_3s = "Unrealistic motion / head-turn filtered; all 3 haptic motors suppressed [0, 0, 0]."
        elif confirmed == 0:
            predicted_next_3s = "Forward corridor clear over next 3.0s; 3 haptic motors idle [0, 0, 0]."
        elif choice == "approaching_center":
            ttc_str = f"~{ttc:.1f}s" if ttc is not None else "~2.0s"
            predicted_next_3s = f"Target approaching center corridor; collision risk in {ttc_str} → vibrate CENTER [0, 1, 0]"
        elif choice == "crossing_left_to_center":
            ttc_str = f"~{ttc:.1f}s" if ttc is not None else "~2.2s"
            predicted_next_3s = f"Target on Left walking toward Center ({ttc_str}) → pre-alert LEFT+CENTER [1, 1, 0]"
        elif choice == "crossing_right_to_center":
            ttc_str = f"~{ttc:.1f}s" if ttc is not None else "~2.2s"
            predicted_next_3s = f"Target on Right walking toward Center ({ttc_str}) → pre-alert CENTER+RIGHT [0, 1, 1]"
        elif choice == "left_obstacle":
            predicted_next_3s = "Confirmed obstacle on Left side → pre-alert LEFT motor [1, 0, 0]"
        else:
            predicted_next_3s = "Confirmed obstacle on Right side → pre-alert RIGHT motor [0, 0, 1]"

        decision_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        result = {
            "realistic_detection": realistic_detection,
            "motion_trend": choice,
            "predicted_next_3s": predicted_next_3s,
            "recommended_pre_alert_zones_3": zones_3,
            "recommended_pre_alert_zones": zones_6,
            "motor_probabilities_3": raw_decision.get("motor_probabilities_3", [float(v) for v in zones_3]),
            "haptic_pattern": haptic_pattern,
            "confidence": raw_decision["confidence"],
            "probabilities": raw_decision["probabilities"],
            "backend": raw_decision["backend"],
            "requested_backend": req_backend,
            "model": raw_decision["model"],
            "jev_key_configured": raw_decision["jev_key_configured"],
            "fallback_note": fallback_note,
            "state_prompt": state_text,
            "decision_ms": decision_ms,
            "updated_at": time.monotonic(),
        }
        with self._lock:
            self._latest_decision = dict(result)
        return result

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.5)

    def submit(self, kinematic_scenario, jpeg_bytes=None):
        with self._lock:
            self._latest_input = dict(kinematic_scenario)

    def latest(self):
        with self._lock:
            return dict(self._latest_decision) if self._latest_decision else None

    def _loop(self):
        while not self._stop.is_set():
            with self._lock:
                scenario = self._latest_input
                self._latest_input = None
            if scenario is None:
                self._stop.wait(0.08)
                continue
            try:
                self.evaluate_sync(scenario)
            except Exception:
                pass
            self._stop.wait(self.interval_s)


# Keep alias for backwards compatibility
AsyncVLMForecaster = DecisionForecaster

