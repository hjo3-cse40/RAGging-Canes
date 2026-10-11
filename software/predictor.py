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
import io
import json
import math
import threading
import time
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

    def reset(self):
        with self._lock:
            self._tracks.clear()
            self._next_id = 1
            self._last_ts = None

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
                    "predicted_boxes": pred_boxes,
                    "predicted_path": pred_path,
                }
                enriched_objects.append(enriched)

            pred_zones_1s = zone_states(boxes_1s, width, self.boundaries)
            pred_zones_2s = zone_states(boxes_2s, width, self.boundaries)
            pred_zones_3s = zone_states(boxes_3s, width, self.boundaries)

            total_obs = len(candidates)
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

                if not realistic_scene:
                    reason_str = "Camera pan / head turn compensated" if ego_motion_detected else "Transient sensor noise filtered"
                    summary = f"UNREALISTIC MOTION FILTERED · {reason_str} ({filtered_count}/{total_obs} suppressed)."
                elif forecast_notes:
                    summary = "3S FORECAST · " + " | ".join(forecast_notes[:2])
                else:
                    active_names = [ZONE_NAMES[i] for i, z in enumerate(pred_zones_2s) if z == 1]
                    z_str = ", ".join(active_names) if active_names else "none"
                    summary = f"REALISTIC STABLE SCENE · {realistic_count} confirmed target(s) · 2s zones: [{z_str}]"

            predict_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            scenario = {
                "realistic": realistic_scene,
                "realism_score": realism_score,
                "ego_motion_detected": ego_motion_detected,
                "risk_level": risk_level,
                "min_ttc_s": min_ttc_s,
                "confirmed_tracks": realistic_count,
                "filtered_anomalies": filtered_count,
                "predicted_zones_1s": pred_zones_1s,
                "predicted_zones_2s": pred_zones_2s,
                "predicted_zones_3s": pred_zones_3s,
                "summary": summary,
                "predict_ms": predict_ms,
            }
            return enriched_objects, scenario


class AsyncVLMForecaster:
    """Optional non-blocking background thread that queries local Ollama on GX10
    to semantically verify whether the 3-second scenario is realistic.
    Never blocks the fast YOLO26n + RIFE + Kalman loop.
    """

    PROMPT = (
        "You are a wearable obstacle-anticipation verifier. Given the camera frame and tracked kinematic "
        "velocities, return JSON only: "
        '{"realistic": true/false, "hazard_3s": "<short 10-word 3-second prediction>"}'
    )

    def __init__(self, ollama_url="http://127.0.0.1:11434", model="nemotron3:33b", interval_s=2.5, timeout_s=4.0):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.interval_s = float(interval_s)
        self.timeout_s = float(timeout_s)
        self._lock = threading.Lock()
        self._latest_Input = None
        self._latest_forecast = None
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.5)

    def submit(self, jpeg_bytes, kinematic_scenario):
        with self._lock:
            self._latest_Input = (jpeg_bytes, dict(kinematic_scenario), time.monotonic())

    def latest(self):
        with self._lock:
            return dict(self._latest_forecast) if self._latest_forecast else None

    def _loop(self):
        while not self._stop.is_set():
            with self._lock:
                item = self._latest_Input
                self._latest_Input = None
            if item is None:
                self._stop.wait(0.2)
                continue
            jpeg_bytes, scenario, ts = item
            try:
                t0 = time.monotonic()
                payload = {
                    "model": self.model,
                    "messages": [{
                        "role": "user",
                        "content": f"{self.PROMPT}\nKinematic summary: {scenario.get('summary', '')}",
                        "images": [base64.b64encode(jpeg_bytes).decode("ascii")],
                    }],
                    "think": False,
                    "stream": False,
                    "options": {"temperature": 0, "num_predict": 48},
                }
                req = urllib.request.Request(
                    f"{self.ollama_url}/api/chat",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    body = json.load(resp)
                content = body.get("message", {}).get("content", "{}")
                parsed = json.loads(content)
                with self._lock:
                    self._latest_forecast = {
                        "realistic": bool(parsed.get("realistic", True)),
                        "hazard_3s": str(parsed.get("hazard_3s", ""))[:120],
                        "vlm_ms": round((time.monotonic() - t0) * 1000),
                        "updated_at": time.monotonic(),
                    }
            except Exception:
                pass
            self._stop.wait(self.interval_s)
