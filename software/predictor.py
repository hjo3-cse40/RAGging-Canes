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


# Physical class priors for Monocular Depth Estimation, Stationary Realism Filtering, and Spatial Memory
CLASS_METADATA = {
    "person": {"height_m": 1.70, "width_m": 0.52, "dynamic": True, "category": "pedestrian"},
    "bicycle": {"height_m": 1.05, "width_m": 0.65, "dynamic": True, "category": "vehicle"},
    "car": {"height_m": 1.50, "width_m": 1.80, "dynamic": True, "category": "vehicle"},
    "motorcycle": {"height_m": 1.15, "width_m": 0.75, "dynamic": True, "category": "vehicle"},
    "dog": {"height_m": 0.50, "width_m": 0.45, "dynamic": True, "category": "pet"},
    "cat": {"height_m": 0.30, "width_m": 0.35, "dynamic": True, "category": "pet"},
    "chair": {"height_m": 0.85, "width_m": 0.55, "dynamic": False, "category": "furniture"},
    "couch": {"height_m": 0.85, "width_m": 1.85, "dynamic": False, "category": "furniture"},
    "bench": {"height_m": 0.80, "width_m": 1.40, "dynamic": False, "category": "furniture"},
    "dining table": {"height_m": 0.76, "width_m": 1.25, "dynamic": False, "category": "furniture"},
    "table": {"height_m": 0.76, "width_m": 1.20, "dynamic": False, "category": "furniture"},
    "desk": {"height_m": 0.75, "width_m": 1.20, "dynamic": False, "category": "furniture"},
    "bed": {"height_m": 0.65, "width_m": 1.50, "dynamic": False, "category": "furniture"},
    "potted plant": {"height_m": 0.75, "width_m": 0.45, "dynamic": False, "category": "obstacle"},
    "refrigerator": {"height_m": 1.75, "width_m": 0.80, "dynamic": False, "category": "structure"},
    "door": {"height_m": 2.00, "width_m": 0.90, "dynamic": False, "category": "structure"},
    "stairs": {"height_m": 0.95, "width_m": 1.20, "dynamic": False, "category": "elevation"},
    "curb": {"height_m": 0.20, "width_m": 1.20, "dynamic": False, "category": "elevation"},
    "trash can": {"height_m": 0.75, "width_m": 0.45, "dynamic": False, "category": "obstacle"},
    "backpack": {"height_m": 0.48, "width_m": 0.36, "dynamic": False, "category": "trip_hazard"},
    "handbag": {"height_m": 0.38, "width_m": 0.34, "dynamic": False, "category": "trip_hazard"},
    "suitcase": {"height_m": 0.62, "width_m": 0.42, "dynamic": False, "category": "trip_hazard"},
    "tv": {"height_m": 0.55, "width_m": 0.95, "dynamic": False, "category": "electronics"},
    "laptop": {"height_m": 0.28, "width_m": 0.36, "dynamic": False, "category": "electronics"},
}


def get_class_metadata(cls_name):
    key = str(cls_name or "person").strip().lower()
    return CLASS_METADATA.get(key, {"height_m": 0.90, "width_m": 0.55, "dynamic": False, "category": "obstacle"})


class MonocularDepthEstimator:
    """GX10 Local Monocular Metric Depth Estimator (Neural Depth-Anything-V2 + Pinhole/Horizon Geometry).

    Converts 2D camera detections into metric (X, Y) floor coordinates in meters on the top-down radar
    even when manual 4-point floor calibration is not present or when an object's feet are cropped.

    Modes:
    - `auto` / `hybrid`: Uses local HuggingFace `depth-anything/Depth-Anything-V2-Small-hf` on GX10 GPU/MPS
      if cached/available in background, fused with pinhole class-height & floor-horizon geometry.
    - `pinhole`: Ultra-fast (<0.2 ms) geometric + physical class-height + horizon-perspective metric depth.
    - `neural`: Prioritizes dense neural depth map from `Depth-Anything-V2` on GPU/MPS.
    - `off`: Disabled (relies strictly on manual 4-point homography calibration).
    """

    DEFAULT_HF_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
    CAMERA_HFOV_DEG = 62.0
    CAMERA_HEIGHT_M = 1.62  # Typical cap-mounted camera height on standing wearer

    def __init__(self, mode="auto", device="auto", model_id=None):
        self.mode = mode if mode in ("auto", "hybrid", "pinhole", "neural", "off") else "auto"
        self.device = RIFEInterpolator._resolve_device(device)
        self.model_id = model_id or self.DEFAULT_HF_MODEL
        self._lock = threading.Lock()
        self._pipe = None
        self._pipe_failed = False
        self._latest_depth_map = None
        self._last_depth_ms = 0.0
        self._active_backend = "pinhole_metric_hybrid" if self.mode != "off" else "off"
        # Asynchronous GX10 VLM Depth Reasoner calibration state
        self._vlm_scale_factor = 1.0
        self._vlm_horizon_ratio = None
        self._vlm_track_depths = {}  # track_id -> {"depth_m": float, "x_m": float, "ts": float}
        self._vlm_spatial_note = None
        self._vlm_model_name = None
        self._vlm_last_ts = 0.0

    def set_mode(self, mode):
        if mode not in ("auto", "hybrid", "pinhole", "neural", "off"):
            raise ValueError("Depth mode must be auto, hybrid, pinhole, neural, or off")
        with self._lock:
            self.mode = mode
            self._active_backend = "off" if mode == "off" else (
                "depth_anything_v2" if (self._pipe is not None and mode in ("auto", "hybrid", "neural"))
                else "pinhole_metric_hybrid"
            )

    _SHARED_PIPES = {}
    _SHARED_PIPE_LOCK = threading.Lock()

    def update_vlm_calibration(self, scale_factor=None, horizon_ratio=None, track_depths=None,
                               reasoning_note=None, model_name="gemma4:26b"):
        """Apply asynchronous depth & perspective calibration reasoned by the GX10 local VLM."""
        now = time.monotonic()
        with self._lock:
            if scale_factor is not None:
                try:
                    sf = max(0.40, min(1.75, float(scale_factor)))
                    self._vlm_scale_factor = round(0.45 * self._vlm_scale_factor + 0.55 * sf, 3)
                except (TypeError, ValueError):
                    pass
            if horizon_ratio is not None:
                try:
                    hr = max(0.04, min(0.55, float(horizon_ratio)))
                    self._vlm_horizon_ratio = round(
                        hr if self._vlm_horizon_ratio is None else (0.45 * self._vlm_horizon_ratio + 0.55 * hr),
                        3,
                    )
                except (TypeError, ValueError):
                    pass
            if isinstance(track_depths, list):
                for item in track_depths:
                    if not isinstance(item, dict):
                        continue
                    tid = item.get("track_id")
                    v_depth = item.get("verified_depth_m", item.get("depth_m"))
                    if tid is not None and v_depth is not None:
                        try:
                            entry = {"depth_m": max(0.4, min(15.0, float(v_depth))), "ts": now}
                            if item.get("verified_x_m") is not None:
                                entry["x_m"] = max(-5.5, min(5.5, float(item["verified_x_m"])))
                            self._vlm_track_depths[int(tid)] = entry
                        except (TypeError, ValueError):
                            pass
            # Prune stale per-track VLM overrides older than 8s
            self._vlm_track_depths = {
                k: v for k, v in self._vlm_track_depths.items() if (now - v.get("ts", 0.0)) <= 8.0
            }
            if reasoning_note:
                self._vlm_spatial_note = str(reasoning_note).strip()
            self._vlm_model_name = model_name
            self._vlm_last_ts = now

    def _ensure_neural_pipe(self):
        if self._pipe is not None:
            return self._pipe
        if self._pipe_failed or self.mode in ("pinhole", "off"):
            return None
        # In 'auto' mode, only run heavy dense transformer depth on GPU (CUDA/MPS) so CPU stays <1ms
        if self.mode == "auto" and self.device == "cpu":
            return None
        cache_key = (self.model_id, self.device)
        with self._SHARED_PIPE_LOCK:
            if cache_key in self._SHARED_PIPES:
                self._pipe = self._SHARED_PIPES[cache_key]
                self._active_backend = "depth_anything_v2"
                return self._pipe
        try:
            os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
            from transformers import pipeline as hf_pipeline
            dev_arg = 0 if self.device.startswith("cuda") else ("mps" if self.device == "mps" else -1)
            # Load from local cache first so we never stall offline test runners
            try:
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                self._pipe = hf_pipeline(
                    task="depth-estimation",
                    model=self.model_id,
                    device=dev_arg,
                )
            except Exception:
                if self.mode == "neural":
                    os.environ.pop("HF_HUB_OFFLINE", None)
                    os.environ.pop("TRANSFORMERS_OFFLINE", None)
                    self._pipe = hf_pipeline(
                        task="depth-estimation",
                        model=self.model_id,
                        device=dev_arg,
                    )
                else:
                    self._pipe_failed = True
                    return None
            with self._SHARED_PIPE_LOCK:
                self._SHARED_PIPES[cache_key] = self._pipe
            self._active_backend = "depth_anything_v2"
            return self._pipe
        except Exception:
            self._pipe_failed = True
            return None

    def _compute_neural_depth_map(self, frame_bgr=None, jpeg_bytes=None):
        valid_jpeg = isinstance(jpeg_bytes, (bytes, bytearray)) and len(jpeg_bytes) > 64 and jpeg_bytes[:2] == b"\xff\xd8"
        if (frame_bgr is None and not valid_jpeg) or self.mode in ("pinhole", "off"):
            return None
        now = time.monotonic()
        with self._lock:
            last_ts = getattr(self, "_latest_depth_ts", 0.0)
            if self._latest_depth_map is not None and (now - last_ts) < 0.12:
                return self._latest_depth_map
        pipe = self._ensure_neural_pipe()
        if pipe is None or np is None:
            return None
        t0 = time.perf_counter()
        try:
            if frame_bgr is not None:
                rgb = frame_bgr[:, :, ::-1] if frame_bgr.ndim == 3 else frame_bgr
                pil_im = Image.fromarray(rgb)
            else:
                pil_im = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
            if pil_im.width > 384:
                pil_im = pil_im.resize((384, 288), Image.Resampling.BILINEAR)
            out = pipe(pil_im)
            depth_tensor = out.get("predicted_depth")
            if depth_tensor is not None:
                d_arr = depth_tensor.detach().cpu().numpy() if hasattr(depth_tensor, "detach") else np.asarray(depth_tensor)
                if d_arr.ndim == 3:
                    d_arr = d_arr[0]
                d_min, d_max = float(np.percentile(d_arr, 2)), float(np.percentile(d_arr, 98))
                if d_max > d_min + 1e-5:
                    norm_inv = np.clip((d_arr - d_min) / (d_max - d_min), 0.02, 1.0)
                    with self._lock:
                        self._latest_depth_map = norm_inv
                        self._latest_depth_ts = time.monotonic()
                        self._last_depth_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                        self._active_backend = "depth_anything_v2"
                    return norm_inv
        except Exception:
            pass
        return None

    def estimate_object(self, obj, width, height, inv_depth_map=None, scene_horizon_ratio=0.30, scene_scale=1.0):
        """Estimate metric distance `depth_m` and `[x_m, y_m]` ground coordinates for one detection."""
        width = max(1, float(width or 640))
        height = max(1, float(height or 480))
        xyxy = obj.get("xyxy", [0, 0, 10, 10])
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        bw = max(6.0, x2 - x1)
        bh = max(8.0, y2 - y1)
        cx = 0.5 * (x1 + x2)

        meta = get_class_metadata(obj.get("class", "person"))
        real_h = meta["height_m"]
        real_w = meta["width_m"]

        # 1. Pinhole focal length from horizontal FOV
        hfov_rad = math.radians(self.CAMERA_HFOV_DEG)
        fx = (0.5 * width) / math.tan(0.5 * hfov_rad)
        fy = fx

        # 2. Angular size depth estimate (compensating for bottom/top cropping and elevated cameras)
        bottom_cropped = y2 >= (height - 6.0)
        top_cropped = y1 <= 6.0
        # Soft-compress extreme far-field box-height blowup when viewing from an elevated/angled camera
        raw_z_height = (fy * real_h) / max(12.0, bh)
        if raw_z_height > 4.2:
            z_height = 4.2 + math.pow(raw_z_height - 4.2, 0.72)
        else:
            z_height = raw_z_height
        z_width = (fx * real_w) / max(10.0, bw)

        if bottom_cropped or top_cropped:
            z_pinhole = 0.45 * z_height + 0.55 * z_width
        else:
            z_pinhole = 0.72 * z_height + 0.28 * z_width

        # 3. Ground-contact vertical horizon perspective (higher y2 = lower in image = closer to camera)
        horizon_y = scene_horizon_ratio * height
        if not bottom_cropped and y2 > horizon_y + 8.0:
            norm_depression = max(0.06, min(1.0, (y2 - horizon_y) / max(1.0, (height - horizon_y))))
            # Maps norm_depression=1.0 (bottom edge) -> ~1.15m, norm_depression=0.50 -> ~2.4m, norm_depression=0.25 -> ~5.0m
            z_floor = 1.15 / math.pow(norm_depression, 1.05)
            # Weight ground-contact perspective more strongly when objects are smaller / elevated camera view
            floor_weight = 0.64 if (bh / height) < 0.30 else 0.42
            z_geom = (1.0 - floor_weight) * z_pinhole + floor_weight * z_floor
        else:
            z_geom = z_pinhole

        # 4. Fuse with dense neural inverse depth map from GX10 Depth-Anything-V2
        z_final = z_geom
        source = "pinhole_metric_hybrid"
        if inv_depth_map is not None and np is not None:
            dh, dw = inv_depth_map.shape[:2]
            # Sample lower-center patch (torso to feet) for accurate ground-contact depth
            ix1 = int(max(0, min(dw - 1, (x1 + 0.22 * bw) * dw / width)))
            ix2 = int(max(ix1 + 1, min(dw, (x2 - 0.22 * bw) * dw / width)))
            iy1 = int(max(0, min(dh - 1, (y1 + 0.35 * bh) * dh / height)))
            iy2 = int(max(iy1 + 1, min(dh, (y2 - 0.08 * bh) * dh / height)))
            patch = inv_depth_map[iy1:iy2, ix1:ix2]
            if patch.size > 0:
                rel_closeness = float(np.median(patch))  # 1.0 = very close (~0.8m), 0.10 = far (~7m)
                z_neural = 0.85 / max(0.11, math.pow(rel_closeness, 0.82))
                neural_w = 0.62 if self.mode == "neural" else 0.52
                z_final = neural_w * z_neural + (1.0 - neural_w) * z_geom
                source = "depth_anything_v2"

        # 5. Apply GX10 Local VLM Depth Reasoner scale calibration & per-track verification
        z_final = z_final * scene_scale
        tid = obj.get("track_id")
        with self._lock:
            vlm_override = self._vlm_track_depths.get(int(tid)) if tid is not None else None
            vlm_model = self._vlm_model_name
        if vlm_override is not None:
            # Smoothly blend 65% VLM-verified depth with 35% live frame depth so motion stays responsive
            z_final = 0.65 * vlm_override["depth_m"] + 0.35 * z_final
            source = f"vlm_verified:{vlm_model or 'gx10'}"

        # Allow physical range up to 14.0m (radar auto-scales its rings to fit all targets inside the dome)
        y_m = max(0.35, min(14.0, z_final))
        # Lateral X (meters right of camera centerline)
        u_offset = (cx / width) - 0.5
        angle_x = u_offset * hfov_rad
        x_m = y_m * math.tan(angle_x)
        if vlm_override is not None and "x_m" in vlm_override:
            x_m = 0.6 * vlm_override["x_m"] + 0.4 * x_m
        x_m = max(-6.0, min(6.0, x_m))
        depth_m = round(math.hypot(x_m, y_m), 2)
        return {
            "depth_m": depth_m,
            "depth_xy": [round(x_m, 3), round(y_m, 3)],
            "depth_source": source,
            "category": meta["category"],
            "dynamic_class": meta["dynamic"],
        }

    def estimate_objects(self, objects, width, height, frame_bgr=None, calibration_config=None, jpeg_bytes=None):
        """Enrich objects with monocular depth (`depth_m`, `depth_xy`) and auto-fill `ground_xy` when uncalibrated."""
        if self.mode == "off":
            return [dict(o) for o in (objects or [])], {"enabled": False, "backend": "off", "depth_ms": 0.0}

        t0 = time.perf_counter()
        inv_map = self._compute_neural_depth_map(frame_bgr=frame_bgr, jpeg_bytes=jpeg_bytes)
        width_f = max(1.0, float(width or 640))
        height_f = max(1.0, float(height or 480))

        # Scene-adaptive horizon & scale: if objects are small or high up in the frame (e.g. elevated/outdoor camera),
        # automatically raise the horizon and normalize scale so targets spread naturally across 2.5m..7.0m
        objs_list = list(objects or [])
        with self._lock:
            vlm_scale = self._vlm_scale_factor
            vlm_horizon = self._vlm_horizon_ratio
            vlm_note = self._vlm_spatial_note
            vlm_model = self._vlm_model_name

        scene_horizon = vlm_horizon if vlm_horizon is not None else 0.30
        auto_scale = vlm_scale
        if objs_list and vlm_horizon is None:
            min_y1_ratio = min(float(o.get("xyxy", [0, 0, 10, 10])[1]) / height_f for o in objs_list)
            med_bh_ratio = sorted(
                (float(o.get("xyxy", [0, 0, 10, 10])[3]) - float(o.get("xyxy", [0, 0, 10, 10])[1])) / height_f
                for o in objs_list
            )[len(objs_list) // 2]
            if min_y1_ratio < 0.35 or med_bh_ratio < 0.26:
                # Elevated or wide-field camera view: horizon is near the top of the frame
                scene_horizon = max(0.06, min(0.24, min_y1_ratio * 0.55))
                if vlm_scale == 1.0 and med_bh_ratio < 0.25:
                    # Adapt scale for elevated/wide perspective so mid-ground targets map to 3.5m..6.2m
                    auto_scale = max(0.58, min(0.92, med_bh_ratio / 0.28))

        enriched = []
        for obj in objs_list:
            item = dict(obj)
            est = self.estimate_object(
                item, width_f, height_f,
                inv_depth_map=inv_map,
                scene_horizon_ratio=scene_horizon,
                scene_scale=auto_scale,
            )
            item["depth_m"] = est["depth_m"]
            item["depth_xy"] = est["depth_xy"]
            item["depth_source"] = est["depth_source"]
            item["category"] = est["category"]
            if item.get("ground_xy") is None and calibration_config is None:
                item["ground_xy"] = est["depth_xy"]
                item["position_kind"] = "monocular_depth"
            elif item.get("ground_xy") is not None and item.get("depth_m") is not None:
                item["depth_m"] = round(math.hypot(item["ground_xy"][0], item["ground_xy"][1]), 2)
            enriched.append(item)

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return enriched, {
            "enabled": True,
            "mode": self.mode,
            "backend": self._active_backend,
            "depth_ms": elapsed_ms,
            "vlm_scale_factor": round(auto_scale, 3),
            "scene_horizon_ratio": round(scene_horizon, 3),
            "vlm_model": vlm_model,
            "vlm_note": vlm_note,
        }


class TrajectoryPredictor:
    """Kalman / Kinematic 3-Second Object Trajectory Predictor, Realism Filter & Persistent Spatial Memory.

    Tracks detected objects (`person`, `chair`, `dining table`, `backpack`, `door`, etc.), evaluates
    whether their motion is physically realistic vs. sensor noise / head-turn ego-motion, maintains a
    multi-second ego-centric Spatial Memory Map across occlusions and head turns, and projects 0.5s–3.0s
    future bounding boxes, ground-plane paths, Time-To-Collision (TTC), and 6-zone states.
    """

    HORIZONS = (0.5, 1.0, 2.0, 3.0)
    MAX_METRIC_SPEED_MPS = 4.5        # Maximum physically realistic indoor human speed (m/s)
    MAX_NORM_SPEED_PER_S = 1.45       # Maximum normalized screen widths/s before flagging teleport/head-whip
    MAX_SCALE_JUMP_PER_S = 2.5        # Maximum relative bounding-box height change rate
    CAMERA_HFOV_RAD = math.radians(62.0)

    def __init__(self, boundaries=None, max_track_idle_s=1.5, memory_ttl_s=12.0,
                 alpha_pos=0.65, alpha_vel=0.45):
        self.boundaries = boundaries or [i / 6 for i in range(7)]
        self.max_track_idle_s = float(max_track_idle_s)
        self.memory_ttl_s = max(0.0, float(memory_ttl_s))
        self.alpha_pos = float(alpha_pos)
        self.alpha_vel = float(alpha_vel)
        self._lock = threading.Lock()
        self._tracks = {}
        self._occupancy_grid = {}
        self._next_id = 1
        self._last_ts = None
        self._keyframes = deque(maxlen=45)
        self._vlm_recommended_range_m = None
        self._vlm_depth_report = None

    def set_memory_ttl(self, memory_ttl_s):
        val = float(memory_ttl_s)
        if not math.isfinite(val) or not (0.0 <= val <= 60.0):
            raise ValueError("memory_ttl_s must be between 0 and 60 seconds")
        with self._lock:
            self.memory_ttl_s = val
            if val <= 0.0:
                self._occupancy_grid.clear()

    def reset(self):
        with self._lock:
            self._tracks.clear()
            self._occupancy_grid.clear()
            self._next_id = 1
            self._last_ts = None
            self._keyframes.clear()
            self._vlm_recommended_range_m = None
            self._vlm_depth_report = None

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

    @staticmethod
    def _rotate_ground_xy(ground_xy, delta_yaw_rad):
        """Rotate ego-centric ground coordinates (x_m, y_m) around wearer origin (0, 0) on head turn."""
        if not ground_xy or abs(delta_yaw_rad) < 1e-4:
            return ground_xy
        gx, gy = float(ground_xy[0]), float(ground_xy[1])
        cos_a = math.cos(delta_yaw_rad)
        sin_a = math.sin(delta_yaw_rad)
        rx = gx * cos_a - gy * sin_a
        ry = gx * sin_a + gy * cos_a
        rx = max(-6.5, min(6.5, rx))
        ry = max(0.15, min(14.5, ry))
        return [round(rx, 3), round(ry, 3)]

    @staticmethod
    def _quantize_voxel(gx, gy):
        """Tesla-style dynamic voxel resolution: 0.35m fine near-field (<2.5m), 0.50m coarse far-field."""
        dist = math.hypot(gx, gy)
        res = 0.35 if dist < 2.5 else 0.50
        return (round(round(gx / res) * res, 2), round(round(gy / res) * res, 2), res)

    def _update_occupancy_cell(self, trk, now, width):
        """Tesla Vector-Space Log-Odds Voxel + Waymo 2D Occupancy Flow Field update."""
        if self.memory_ttl_s <= 0.0:
            return
        gxy = trk.get("ground_xy")
        if gxy:
            gx, gy = float(gxy[0]), float(gxy[1])
        else:
            u = (trk["cx"] / max(1.0, float(width))) - 0.5
            ang = u * self.CAMERA_HFOV_RAD
            dist = float(trk.get("depth_m") or 2.8)
            gx = dist * math.sin(ang)
            gy = dist * math.cos(ang)
        qx, qy, res = self._quantize_voxel(gx, gy)
        cell_key = (qx, qy)
        existing = self._occupancy_grid.get(cell_key)
        meta = get_class_metadata(trk["class"])
        prev_log_odds = existing["log_odds"] if existing else 0.0
        # Bayesian log-odds increment on observation
        new_log_odds = min(4.2, prev_log_odds + (1.35 if trk["confidence"] >= 0.6 else 0.85))
        occ_prob = round(1.0 / (1.0 + math.exp(-new_log_odds)), 3)

        # Waymo Occupancy Flow vector (m/s in BEV ground space)
        f_vx = float(trk.get("gvx", 0.0))
        f_vy = float(trk.get("gvy", 0.0))
        if not gxy:
            # Approximate metric flow from pixel velocity
            f_vx = round((trk.get("vx", 0.0) / max(1.0, float(width))) * 2.5, 2)
            f_vy = round(-(trk.get("vy", 0.0) / 480.0) * 2.0, 2)
        if existing:
            f_vx = round(0.5 * existing.get("flow_vx", 0.0) + 0.5 * f_vx, 2)
            f_vy = round(0.5 * existing.get("flow_vy", 0.0) + 0.5 * f_vy, 2)

        self._occupancy_grid[cell_key] = {
            "x": qx,
            "y": qy,
            "x_m": qx,
            "y_m": qy,
            "res_m": res,
            "class": trk["class"],
            "category": meta["category"],
            "dynamic": meta["dynamic"],
            "hits": (existing["hits"] + 1) if existing else 1,
            "log_odds": round(new_log_odds, 3),
            "occ_prob": occ_prob,
            "flow_vx": round(f_vx, 2),
            "flow_vy": round(f_vy, 2),
            "confidence": round(max(trk["confidence"], existing["confidence"] if existing else 0.0), 3),
            "last_seen": now,
            "in_shadow": False,
            "source": trk.get("depth_verified_by") or "yolo",
        }

    def apply_vlm_depth_verification(self, track_depths=None, recommended_radar_range_m=None,
                                     scene_scale_factor=None, scale_factor=None, horizon_ratio=None,
                                     reasoning_note=None, depth_reasoning=None, model_name="gemma4:26b"):
        """Update active/remembered tracks and the BEV occupancy grid with GX10 VLM depth verification."""
        eff_scale = scene_scale_factor if scene_scale_factor is not None else scale_factor
        eff_reason = reasoning_note if reasoning_note is not None else depth_reasoning
        with self._lock:
            now = self._last_ts if self._last_ts is not None else time.monotonic()
            if recommended_radar_range_m is not None:
                try:
                    rr = max(5.0, min(15.0, float(recommended_radar_range_m)))
                    self._vlm_recommended_range_m = round(
                        rr if self._vlm_recommended_range_m is None else (0.4 * self._vlm_recommended_range_m + 0.6 * rr),
                        2,
                    )
                except (TypeError, ValueError):
                    pass

            corrected_count = 0
            if isinstance(track_depths, list):
                for item in track_depths:
                    if not isinstance(item, dict):
                        continue
                    tid = item.get("track_id")
                    v_depth = item.get("verified_depth_m", item.get("depth_m"))
                    if tid is None or v_depth is None:
                        continue
                    try:
                        tid_int = int(tid)
                        dist_m = max(0.4, min(14.0, float(v_depth)))
                    except (TypeError, ValueError):
                        continue
                    trk = self._tracks.get(tid_int)
                    if trk is None:
                        continue
                    old_gxy = trk.get("ground_xy")
                    if item.get("verified_x_m") is not None:
                        try:
                            gx = max(-5.5, min(5.5, float(item["verified_x_m"])))
                        except (TypeError, ValueError):
                            gx = float(old_gxy[0]) if old_gxy else 0.0
                    elif old_gxy:
                        old_dist = max(0.25, math.hypot(old_gxy[0], old_gxy[1]))
                        gx = float(old_gxy[0]) * (dist_m / old_dist)
                    else:
                        u = (trk["cx"] / 640.0) - 0.5
                        gx = dist_m * math.sin(u * self.CAMERA_HFOV_RAD)
                    gy = max(0.35, math.sqrt(max(0.15, dist_m * dist_m - gx * gx)))
                    # Blend VLM-verified position into track & BEV voxel grid
                    if old_gxy:
                        gx = 0.65 * gx + 0.35 * float(old_gxy[0])
                        gy = 0.65 * gy + 0.35 * float(old_gxy[1])
                    trk["ground_xy"] = [round(gx, 3), round(gy, 3)]
                    trk["depth_m"] = round(math.hypot(gx, gy), 2)
                    trk["depth_verified_by"] = model_name
                    self._update_occupancy_cell(trk, now, 640)
                    corrected_count += 1

            self._vlm_depth_report = {
                "active": True,
                "model": model_name,
                "model_name": model_name,
                "scale_factor": eff_scale if eff_scale is not None else 1.0,
                "scene_scale_factor": eff_scale if eff_scale is not None else 1.0,
                "horizon_ratio": horizon_ratio,
                "recommended_radar_range_m": self._vlm_recommended_range_m,
                "corrected_tracks": corrected_count,
                "reasoning": str(eff_reason or "").strip(),
                "updated_at": time.monotonic(),
            }

    def inject_vlm_obstacles(self, vlm_obstacles, now=None):
        """Inject open-world obstacles discovered by GX10 Ollama VLM (gemma4:26b / qwen3.6:35b / nemotron3:33b)
        directly into the persistent BEV occupancy grid."""
        if not vlm_obstacles or self.memory_ttl_s <= 0.0:
            return
        with self._lock:
            ts = float(now) if now is not None else (self._last_ts if self._last_ts is not None else time.monotonic())
            zone_x_map = {"LEFT": -1.1, "CENTER": 0.0, "RIGHT": 1.1}
            for item in vlm_obstacles[:6]:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or item.get("label") or item.get("class") or "obstacle").strip().lower()[:24]
                zone = str(item.get("zone") or "CENTER").strip().upper()
                try:
                    dist_m = max(0.5, min(12.0, float(item.get("distance_m", 2.2))))
                except (TypeError, ValueError):
                    dist_m = 2.2
                base_x = zone_x_map.get(zone, 0.0) * max(0.6, min(2.4, dist_m * 0.35))
                gy = max(0.35, math.sqrt(max(0.15, dist_m * dist_m - base_x * base_x)))
                qx, qy, res = self._quantize_voxel(base_x, gy)
                meta = get_class_metadata(name)
                cell_key = (qx, qy)
                existing = self._occupancy_grid.get(cell_key)
                lo = min(3.8, (existing["log_odds"] if existing else 0.0) + 1.15)
                occ_p = round(1.0 / (1.0 + math.exp(-lo)), 3)
                self._occupancy_grid[cell_key] = {
                    "x": qx,
                    "y": qy,
                    "x_m": qx,
                    "y_m": qy,
                    "res_m": res,
                    "class": name,
                    "category": meta["category"],
                    "dynamic": meta["dynamic"],
                    "hits": (existing["hits"] + 1) if existing else 1,
                    "log_odds": round(lo, 3),
                    "occ_prob": occ_p,
                    "flow_vx": existing.get("flow_vx", 0.0) if existing else 0.0,
                    "flow_vy": existing.get("flow_vy", 0.0) if existing else 0.0,
                    "confidence": 0.85,
                    "last_seen": ts,
                    "in_shadow": True,
                    "source": "gx10_vlm",
                }

    def _step_bev_occupancy_and_raycast(self, enriched_objects, width, now, dt_global, delta_yaw_rad):
        """Waymo Raycast Free-Space Clearing + Occlusion Shadowing + Tesla Drivable/Walkable Corridor."""
        max_range_m = 6.5
        num_rays = 31
        min_ang = -math.radians(75.0)
        max_ang = math.radians(75.0)
        half_fov = 0.5 * self.CAMERA_HFOV_RAD

        # 1. Ego-motion rotation & flow advection on existing BEV voxels
        occ_ttl = max(15.0, self.memory_ttl_s * 1.25) if self.memory_ttl_s > 0 else 0.0
        updated_grid = {}
        for _, cell in list(self._occupancy_grid.items()):
            age = now - cell["last_seen"]
            if age > occ_ttl or self.memory_ttl_s <= 0:
                continue
            cx_m, cy_m = cell["x"], cell["y"]
            if abs(delta_yaw_rad) > 1e-3:
                cx_m, cy_m = self._rotate_ground_xy([cx_m, cy_m], delta_yaw_rad)
            # Advect dynamic flow cells slightly when recently unseen
            if cell.get("dynamic") and 0.05 < age <= 1.8:
                damp = math.exp(-age * 1.2)
                cx_m += cell.get("flow_vx", 0.0) * dt_global * damp
                cy_m = max(0.2, cy_m + cell.get("flow_vy", 0.0) * dt_global * damp)
            qx, qy, res = self._quantize_voxel(cx_m, cy_m)
            # Mild temporal log-odds decay
            decayed_lo = cell["log_odds"] - (0.08 * dt_global if not cell.get("dynamic") else 0.22 * dt_global)
            if decayed_lo > -0.4:
                occ_p = round(1.0 / (1.0 + math.exp(-decayed_lo)), 3)
                updated_grid[(qx, qy)] = {
                    **cell,
                    "x": qx,
                    "y": qy,
                    "x_m": qx,
                    "y_m": qy,
                    "res_m": res,
                    "log_odds": round(decayed_lo, 3),
                    "occ_prob": occ_p,
                }
        self._occupancy_grid = updated_grid

        # 2. Compute live obstacle polar intervals (angle_rad, dist_m) for raycasting
        live_Polar = []
        for obj in enriched_objects:
            if not obj.get("realistic", True):
                continue
            gxy = obj.get("ground_xy")
            if gxy:
                ox, oy = float(gxy[0]), float(gxy[1])
                ang = math.atan2(ox, max(0.15, oy))
                dist = math.hypot(ox, oy)
                half_w_ang = math.atan2(0.5 * get_class_metadata(obj["class"])["width_m"], max(0.3, dist))
            else:
                x1, _, x2, _ = obj.get("xyxy", [0, 0, 10, 10])
                u_mid = ((x1 + x2) * 0.5 / max(1.0, float(width))) - 0.5
                ang = u_mid * self.CAMERA_HFOV_RAD
                dist = float(obj.get("depth_m") or 2.8)
                half_w_ang = max(0.05, ((x2 - x1) / max(1.0, float(width))) * self.CAMERA_HFOV_RAD * 0.5)
            live_Polar.append((ang - half_w_ang, ang + half_w_ang, dist))

        # 3. Waymo Raycasting: clear cells in front of first hit inside camera FOV; mark cells behind hit as shadow
        for key, cell in list(self._occupancy_grid.items()):
            age = now - cell["last_seen"]
            if age < 0.05:
                cell["in_shadow"] = False
                continue
            cell_ang = math.atan2(cell["x"], max(0.15, cell["y"]))
            cell_dist = math.hypot(cell["x"], cell["y"])
            if abs(cell_ang) <= half_fov:
                # Inside active camera frustum: check if occluded by a closer live detection
                blocking_dist = None
                for a_min, a_max, d_hit in live_Polar:
                    if (a_min - 0.06) <= cell_ang <= (a_max + 0.06):
                        if blocking_dist is None or d_hit < blocking_dist:
                            blocking_dist = d_hit
                if blocking_dist is not None and cell_dist > blocking_dist + 0.25:
                    # Cell is in occlusion shadow behind a foreground object -> preserve in memory!
                    cell["in_shadow"] = True
                elif blocking_dist is None or cell_dist < blocking_dist - 0.40:
                    # Direct unobstructed line-of-sight sees free space where this old cell was
                    cell["in_shadow"] = False
                    # Dynamic objects clear quickly when line-of-sight is empty; static furniture clears gradually
                    clear_step = 0.95 if cell.get("dynamic") else 0.28
                    cell["log_odds"] = round(cell["log_odds"] - clear_step, 3)
                    cell["occ_prob"] = round(1.0 / (1.0 + math.exp(-cell["log_odds"])), 3)
                    if cell["log_odds"] < -0.2:
                        del self._occupancy_grid[key]
            else:
                # Outside current camera frustum (peripheral memory after head turn)
                cell["in_shadow"] = True

        # 4. Build Tesla-style Walkable Free-Space Contour across [-75°, +75°]
        contour = []
        for i in range(num_rays):
            ray_ang = min_ang + (max_ang - min_ang) * (i / float(max(1, num_rays - 1)))
            in_fov = abs(ray_ang) <= half_fov
            ray_range = max_range_m
            ray_state = "clear" if in_fov else "unobserved"

            for a_min, a_max, d_hit in live_Polar:
                if (a_min - 0.04) <= ray_ang <= (a_max + 0.04) and d_hit < ray_range:
                    ray_range = max(0.35, d_hit)
                    ray_state = "obstacle"

            for cell in self._occupancy_grid.values():
                c_ang = math.atan2(cell["x"], max(0.15, cell["y"]))
                c_dist = math.hypot(cell["x"], cell["y"])
                c_half_w = math.atan2(0.5 * cell.get("res_m", 0.4), max(0.3, c_dist))
                if (c_ang - c_half_w - 0.03) <= ray_ang <= (c_ang + c_half_w + 0.03) and c_dist < ray_range:
                    ray_range = max(0.35, c_dist)
                    if ray_state != "obstacle":
                        ray_state = "memory"

            rx = round(ray_range * math.sin(ray_ang), 3)
            ry = round(ray_range * math.cos(ray_ang), 3)
            contour.append({
                "angle_deg": round(math.degrees(ray_ang), 1),
                "range_m": round(ray_range, 2),
                "x": rx,
                "y": ry,
                "state": ray_state,
            })
        return contour

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
                cls_name = obj.get("class", "person")
                meta = get_class_metadata(cls_name)
                candidates.append({
                    "orig_idx": idx,
                    "obj": dict(obj),
                    "class": cls_name,
                    "category": obj.get("category") or meta["category"],
                    "dynamic": meta["dynamic"],
                    "confidence": float(obj.get("confidence", 0.5)),
                    "xyxy": [float(v) for v in xyxy],
                    "cx": cx,
                    "cy": cy,
                    "w": bw,
                    "h": bh,
                    "ground_xy": ground_xy,
                    "depth_m": obj.get("depth_m"),
                })

            # 2. Associate candidates with active or remembered tracks (IoU + normalized center/ground distance)
            matched_tracks = set()
            matched_cands = {}
            pairs = []
            for c_idx, cand in enumerate(candidates):
                for t_id, trk in self._tracks.items():
                    if trk["class"] != cand["class"]:
                        continue
                    dt = max(0.015, min(2.0, now - trk["last_seen"]))
                    pred_cx = trk["cx"] + trk["vx"] * min(dt, 1.5)
                    pred_cy = trk["cy"] + trk["vy"] * min(dt, 1.5)
                    pred_box = self._state_to_box(pred_cx, pred_cy, trk["w"], trk["h"], width, height)
                    iou = self._iou(cand["xyxy"], pred_box)
                    norm_dist = math.hypot((cand["cx"] - pred_cx) / width, (cand["cy"] - pred_cy) / height)
                    ground_bonus = 0.0
                    if cand["ground_xy"] and trk.get("ground_xy"):
                        g_dist = math.hypot(
                            cand["ground_xy"][0] - trk["ground_xy"][0],
                            cand["ground_xy"][1] - trk["ground_xy"][1],
                        )
                        if g_dist < 1.1:
                            ground_bonus = 0.25 * (1.1 - g_dist)
                    if iou > 0.05 or norm_dist < 0.42 or ground_bonus > 0.05:
                        score = iou * 2.0 - norm_dist + ground_bonus
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
            elif len(raw_shifts_vx) == 1:
                # Single stationary furniture/structure shifting rapidly also indicates camera pan
                only_c_idx = next(iter(matched_cands.keys()))
                if not candidates[only_c_idx]["dynamic"] and abs(raw_shifts_vx[0]) / width > 0.35:
                    ego_vx = raw_shifts_vx[0] * 0.7

            # Angular head-turn yaw delta (radians)
            delta_yaw_rad = -(ego_vx * dt_global / float(width)) * self.CAMERA_HFOV_RAD

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

                    # Stationary class prior: furniture/structures shouldn't walk across the room on their own
                    if not cand["dynamic"] and not ego_motion_detected:
                        if (metric_speed is not None and metric_speed > 1.2) or norm_speed > 0.65:
                            velocity_plausible = False

                    if velocity_plausible and scale_coherent:
                        vel_gain = self.alpha_vel if cand["dynamic"] else (self.alpha_vel * 0.35)
                        trk["vx"] = (1.0 - vel_gain) * trk["vx"] + vel_gain * inst_vx
                        trk["vy"] = (1.0 - vel_gain) * trk["vy"] + vel_gain * inst_vy
                        trk["vw"] = (1.0 - vel_gain) * trk["vw"] + vel_gain * inst_vw
                        trk["vh"] = (1.0 - vel_gain) * trk["vh"] + vel_gain * inst_vh
                        trk["cx"] = (1.0 - self.alpha_pos) * (trk["cx"] + trk["vx"] * dt) + self.alpha_pos * cand["cx"]
                        trk["cy"] = (1.0 - self.alpha_pos) * (trk["cy"] + trk["vy"] * dt) + self.alpha_pos * cand["cy"]
                        trk["w"] = (1.0 - self.alpha_pos) * trk["w"] + self.alpha_pos * cand["w"]
                        trk["h"] = (1.0 - self.alpha_pos) * trk["h"] + self.alpha_pos * cand["h"]
                        if cand["ground_xy"]:
                            if trk.get("ground_xy"):
                                trk["gvx"] = (1.0 - vel_gain) * trk.get("gvx", 0.0) + vel_gain * inst_gvx
                                trk["gvy"] = (1.0 - vel_gain) * trk.get("gvy", 0.0) + vel_gain * inst_gvy
                            trk["ground_xy"] = cand["ground_xy"]
                        if dt > 0.65 and cand["confidence"] < 0.55 and trk["hits"] < 2:
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
                    trk["depth_m"] = cand["depth_m"] or trk.get("depth_m")
                    trk["category"] = cand["category"]
                    trk["dynamic"] = cand["dynamic"]
                    trk["last_seen"] = now
                    trk["norm_speed"] = round(math.hypot(trk["vx"] / width, trk["vy"] / height), 3)
                    trk["metric_speed_mps"] = round(
                        math.hypot(trk.get("gvx", 0.0), trk.get("gvy", 0.0)), 2
                    ) if trk.get("ground_xy") else None
                    if trk["realistic"] and (trk["hits"] >= 2 or trk["confidence"] >= 0.55):
                        trk["confirmed_ever"] = True
                        self._update_occupancy_cell(trk, now, width)
                    cand_to_track[c_idx] = trk
                else:
                    t_id = self._next_id
                    self._next_id += 1
                    confirmed_immediately = cand["confidence"] >= 0.55
                    trk = {
                        "track_id": t_id,
                        "class": cand["class"],
                        "category": cand["category"],
                        "dynamic": cand["dynamic"],
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
                        "depth_m": cand["depth_m"],
                        "gvx": 0.0,
                        "gvy": 0.0,
                        "hits": 1,
                        "anomalies": 0,
                        "created_at": now,
                        "last_seen": now,
                        "realistic": confirmed_immediately,
                        "confirmed_ever": confirmed_immediately,
                        "realism_reason": "initial_observation" if confirmed_immediately else "transient_unconfirmed",
                        "norm_speed": 0.0,
                        "metric_speed_mps": 0.0 if cand["ground_xy"] else None,
                    }
                    if confirmed_immediately:
                        self._update_occupancy_cell(trk, now, width)
                    self._tracks[t_id] = trk
                    cand_to_track[c_idx] = trk

            # Prune expired tracks: confirmed tracks persist up to `memory_ttl_s`; unconfirmed blips expire at `max_track_idle_s`
            effective_mem_ttl = max(self.max_track_idle_s, self.memory_ttl_s)
            expired = []
            for tid, t in self._tracks.items():
                idle_s = now - t["last_seen"]
                ttl_limit = effective_mem_ttl if (t.get("confirmed_ever") and self.memory_ttl_s > 0.0) else self.max_track_idle_s
                if idle_s > ttl_limit:
                    expired.append(tid)
            for tid in expired:
                del self._tracks[tid]

            # 5. Roll out 0.5s - 3.0s future horizons for each currently visible object
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
                    "category": trk.get("category", "obstacle"),
                    "hits": trk["hits"],
                    "realistic": is_realistic,
                    "remembered": False,
                    "in_memory": False,
                    "memory_age_s": 0.0,
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

            # 6. Update & project Persistent Spatial Memory tracks (confirmed objects currently occluded or out of FOV)
            remembered_objects = []
            memory_boxes = []
            if self.memory_ttl_s > 0.0:
                for t_id, trk in self._tracks.items():
                    if t_id in matched_tracks or not trk.get("confirmed_ever"):
                        continue
                    mem_age_s = round(now - trk["last_seen"], 2)
                    if mem_age_s <= 0.0 or mem_age_s > self.memory_ttl_s:
                        continue

                    # Apply ego-motion head-turn compensation & gentle coasting for dynamic targets
                    coast_factor = math.exp(-mem_age_s * 1.4) if trk.get("dynamic") else 0.0
                    trk["cx"] = trk["cx"] - ego_vx * dt_global + trk["vx"] * dt_global * coast_factor
                    trk["cy"] = trk["cy"] + trk["vy"] * dt_global * coast_factor
                    trk["vx"] *= (0.85 if trk.get("dynamic") else 0.2)
                    trk["vy"] *= (0.85 if trk.get("dynamic") else 0.2)

                    if trk.get("ground_xy"):
                        rotated_gxy = self._rotate_ground_xy(trk["ground_xy"], delta_yaw_rad)
                        if trk.get("dynamic") and mem_age_s <= 2.0:
                            rotated_gxy = [
                                round(rotated_gxy[0] + trk.get("gvx", 0.0) * dt_global * coast_factor, 3),
                                round(max(0.15, rotated_gxy[1] + trk.get("gvy", 0.0) * dt_global * coast_factor), 3),
                            ]
                        trk["ground_xy"] = rotated_gxy
                        trk["depth_m"] = round(math.hypot(rotated_gxy[0], rotated_gxy[1]), 2)

                    decay = math.exp(-mem_age_s / max(1.0, 0.65 * self.memory_ttl_s))
                    mem_conf = round(max(0.15, trk["confidence"] * decay), 3)
                    mem_box = self._state_to_box(trk["cx"], trk["cy"], trk["w"], trk["h"], width, height)
                    mem_z_idx = int(max(0, min(5, (trk["cx"] / width) * 6)))
                    mem_z_name = ZONE_NAMES[mem_z_idx]
                    mem_z3_name = THREE_ZONE_NAMES[mem_z_idx // 2]

                    if mem_age_s <= 4.0:
                        memory_boxes.append((mem_box[0], mem_box[2]))

                    remembered_objects.append({
                        "track_id": trk["track_id"],
                        "class": trk["class"],
                        "category": trk.get("category", "obstacle"),
                        "confidence": trk["confidence"],
                        "memory_confidence": mem_conf,
                        "xyxy": mem_box,
                        "ground_xy": trk.get("ground_xy"),
                        "depth_m": trk.get("depth_m"),
                        "position_kind": "spatial_memory",
                        "hits": trk["hits"],
                        "realistic": True,
                        "remembered": True,
                        "in_memory": True,
                        "memory_age_s": mem_age_s,
                        "memory_ttl_s": self.memory_ttl_s,
                        "realism_reason": "spatial_memory_persistence",
                        "velocity_px_s": [round(trk["vx"], 1), round(trk["vy"], 1)],
                        "norm_speed": trk["norm_speed"],
                        "metric_speed_mps": trk["metric_speed_mps"],
                        "approaching": False,
                        "ttc_s": None,
                        "current_zone": mem_z_name,
                        "predicted_zone_2s": mem_z_name,
                        "current_zone_3": mem_z3_name,
                        "predicted_zone_3_2s": mem_z3_name,
                        "predicted_boxes": {},
                        "predicted_path": [],
                    })

            # 7. Step Tesla BEV Log-Odds Grid + Waymo Occupancy Flow & Raycast Free-Space Clearing
            free_space_contour = self._step_bev_occupancy_and_raycast(
                enriched_objects, width, now, dt_global, delta_yaw_rad
            )

            pred_zones_1s = zone_states(boxes_1s, width, self.boundaries)
            pred_zones_2s = zone_states(boxes_2s, width, self.boundaries)
            pred_zones_3s = zone_states(boxes_3s, width, self.boundaries)

            curr_boxes = [(c["xyxy"][0], c["xyxy"][2]) for c in candidates]
            curr_zones_6 = zone_states(curr_boxes, width, self.boundaries)
            curr_zones_3 = six_to_three_zones(curr_zones_6)
            pred_zones_3_1s = six_to_three_zones(pred_zones_1s)
            pred_zones_3_2s = six_to_three_zones(pred_zones_2s)
            pred_zones_3_3s = six_to_three_zones(pred_zones_3s)

            mem_zones_6 = zone_states(curr_boxes + memory_boxes, width, self.boundaries)
            mem_zones_3 = six_to_three_zones(mem_zones_6)

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
                if remembered_objects:
                    mem_summary = ", ".join(
                        f"{r['class']} ({r['current_zone']}, {r['memory_age_s']:.1f}s ago)"
                        for r in remembered_objects[:3]
                    )
                    summary = f"SPATIAL MEMORY ACTIVE · {len(remembered_objects)} remembered obstacle(s): {mem_summary}"
                else:
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
                    mem_suffix = f" · +{len(remembered_objects)} in memory" if remembered_objects else ""
                    summary = "3S FORECAST · " + " | ".join(forecast_notes[:2]) + mem_suffix
                else:
                    active_names_3 = [THREE_ZONE_NAMES[i] for i, z in enumerate(pred_zones_3_2s) if z == 1]
                    z_str = ", ".join(active_names_3) if active_names_3 else "none"
                    mem_suffix = f" (+{len(remembered_objects)} in spatial memory)" if remembered_objects else ""
                    summary = f"REALISTIC STABLE SCENE · {realistic_count} confirmed target(s){mem_suffix} · 3-motor zones: [{z_str}]"

            rec_3 = [
                1 if (pred_zones_3_1s[i] == 1 or pred_zones_3_2s[i] == 1) else 0
                for i in range(3)
            ] if realistic_scene else [0, 0, 0]
            rec_6 = [
                1 if (pred_zones_1s[i] == 1 or pred_zones_2s[i] == 1) else 0
                for i in range(6)
            ] if realistic_scene else [0, 0, 0, 0, 0, 0]

            occupancy_cells = [
                {
                    **cell,
                    "age_s": round(now - cell["last_seen"], 2),
                }
                for cell in self._occupancy_grid.values()
            ]

            # Build compact per-track geometry table for the GX10 Local Model Depth Reasoner
            active_tracks_summary = []
            max_observed_dist_m = 4.5
            for obj in enriched_objects:
                if not obj.get("realistic", True):
                    continue
                xyxy = obj.get("xyxy", [0, 0, 10, 10])
                bh = max(4.0, float(xyxy[3]) - float(xyxy[1]))
                dm = obj.get("depth_m")
                if dm is not None:
                    max_observed_dist_m = max(max_observed_dist_m, float(dm))
                active_tracks_summary.append({
                    "track_id": obj.get("track_id"),
                    "class": obj.get("class", "person"),
                    "xyxy": [round(float(v), 1) for v in xyxy],
                    "foot_y_pct": round(float(xyxy[3]) / max(1.0, float(height)), 2),
                    "box_h_pct": round(bh / max(1.0, float(height)), 2),
                    "fast_depth_m": dm,
                    "ground_xy": obj.get("ground_xy"),
                    "zone_3": obj.get("current_zone_3", "CENTER"),
                })
            for rem in remembered_objects:
                dm = rem.get("depth_m")
                if dm is not None:
                    max_observed_dist_m = max(max_observed_dist_m, float(dm))

            # Dynamic radar max range (meters): adapts to farthest target or VLM-verified scene scale
            auto_range_m = max(7.0, min(14.0, math.ceil((max_observed_dist_m + 1.2) * 2.0) / 2.0))
            if self._vlm_recommended_range_m is not None:
                radar_max_range_m = round(max(auto_range_m, self._vlm_recommended_range_m), 1)
            else:
                radar_max_range_m = round(auto_range_m, 1)

            predict_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            scenario = {
                "realistic": realistic_scene,
                "realistic_detection": realistic_scene,
                "realism_score": realism_score,
                "ego_motion_detected": ego_motion_detected,
                "ego_yaw_delta_deg": round(math.degrees(delta_yaw_rad), 2),
                "risk_level": risk_level,
                "motion_trend": motion_trend,
                "min_ttc_s": min_ttc_s,
                "confirmed_tracks": realistic_count,
                "filtered_anomalies": filtered_count,
                "remembered_count": len(remembered_objects),
                "remembered_objects": remembered_objects,
                "memory_ttl_s": self.memory_ttl_s,
                "occupancy_cells": occupancy_cells,
                "free_space_contour": free_space_contour,
                "radar_max_range_m": radar_max_range_m,
                "active_tracks_summary": active_tracks_summary,
                "vlm_depth_report": dict(self._vlm_depth_report) if self._vlm_depth_report else None,
                "zones_3": curr_zones_3,
                "memory_zones_3": mem_zones_3,
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


GX10_OLLAMA_MODELS = (
    "gemma4:26b",
    "qwen3.6:35b",
    "nemotron3:33b",
    "nemotron-3-super:latest",
)


class DecisionForecaster:
    """Asynchronous System-1 Decision Layer + System-2 GX10 Depth & Spatial Map Reasoner.

    Supports:
    - `gemma4:26b` / `qwen3.6:35b` / `nemotron3:33b` / `nemotron-3-super:latest`: Local GX10 Ollama
      Vision/Language Models running on the 128 GB RAM GX10 GPU (`http://127.0.0.1:11434`).
      Also performs asynchronous Depth & Spatial Map Verification to double-check fast monocular
      depths and update the BEV radar map.
    - `laya`: Local open-source `convaiinnovations/laya` decision model (Apache 2.0, ~421M ModernBERT
      with bidirectional option-order de-biasing, ~50–100 ms on MPS/CUDA, 100% offline).
    - `jev`: Hosted TypeSafe AI `POST https://api.typesafe.ai/v1/systemone` endpoint using the key in
      `private/jev_api_key.txt` or `JEV_API_KEY` environment variable.
    - `kinematic`: Zero-weight deterministic fallback.

    Runs in a non-blocking background thread in live mode so it never blocks YOLO.
    """

    LAYA_MODEL_ID = "convaiinnovations/laya"
    LAYA_REVISION = "7b928d828b7b0e022f929d9bd2e44165aa270148"
    JEV_API_URL = "https://api.typesafe.ai/v1/systemone"
    JEV_MODEL = "jev-latest"
    DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"

    # Models installed on the 128 GB RAM ASUS Ascent GX10 (`ollama list`)
    GX10_OLLAMA_MODELS = GX10_OLLAMA_MODELS
    GX10_VISION_OLLAMA_MODELS = (
        "gemma4:26b",
        "qwen3.6:35b",
        "nemotron3:33b",
    )
    VALID_BACKENDS = (
        "auto",
        "laya",
        "jev",
        "kinematic",
        "ollama",
        "gemma4:26b",
        "qwen3.6:35b",
        "nemotron3:33b",
        "nemotron-3-super:latest",
    )

    OLLAMA_SCHEMA = {
        "type": "object",
        "properties": {
            "hazard_choice": {
                "type": "string",
                "enum": [
                    "approaching_center",
                    "crossing_left_to_center",
                    "crossing_right_to_center",
                    "left_obstacle",
                    "right_obstacle",
                    "unrealistic_or_clear",
                ],
            },
            "confidence": {"type": "number"},
            "scene_scale_factor": {"type": "number"},
            "horizon_ratio": {"type": "number"},
            "recommended_radar_range_m": {"type": "number"},
            "track_depths": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "track_id": {"type": "integer"},
                        "verified_depth_m": {"type": "number"},
                        "verified_x_m": {"type": "number"},
                    },
                    "required": ["track_id", "verified_depth_m"],
                },
            },
            "scene_obstacles": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "zone": {"type": "string", "enum": ["LEFT", "CENTER", "RIGHT"]},
                        "distance_m": {"type": "number"},
                    },
                    "required": ["name", "zone", "distance_m"],
                },
            },
            "depth_reasoning": {"type": "string"},
            "guidance": {"type": "string"},
        },
        "required": ["hazard_choice", "confidence", "scene_obstacles", "guidance"],
    }

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
                 jev_url=None, jev_model=None, ollama_url=None, on_vlm_obstacles=None,
                 on_vlm_depth_calibration=None, enable_bg_depth_verifier=False,
                 depth_verifier_model="gemma4:26b",
                 interval_s=0.35, timeout_s=2.5, ollama_timeout_s=15.0):
        self.requested_backend = backend
        self.device = device
        self.jev_key_file = Path(jev_key_file) if jev_key_file else None
        self.jev_url = (jev_url or os.environ.get("JEV_API_URL") or self.JEV_API_URL).rstrip("/")
        self.jev_model = (jev_model or os.environ.get("JEV_MODEL") or self.JEV_MODEL).strip()
        self.ollama_url = (ollama_url or os.environ.get("OLLAMA_URL") or self.DEFAULT_OLLAMA_URL).rstrip("/")
        self.on_vlm_obstacles = on_vlm_obstacles
        self.on_vlm_depth_calibration = on_vlm_depth_calibration
        self.enable_bg_depth_verifier = bool(enable_bg_depth_verifier)
        self.depth_verifier_model = depth_verifier_model
        self.interval_s = float(interval_s)
        self.timeout_s = float(timeout_s)
        self.ollama_timeout_s = float(ollama_timeout_s)
        self._lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._laya_agent = None
        self._laya_error = None
        self._latest_input = None
        self._latest_jpeg = None
        self._latest_decision = None
        self._latest_depth_verification = None
        self._stop = threading.Event()
        self._thread = None
        self._depth_thread = None

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
        if backend not in self.VALID_BACKENDS:
            raise ValueError(f"Decision backend must be one of {self.VALID_BACKENDS}")
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

    def _run_ollama(self, model_name, state_text, jpeg_bytes=None, scenario=None):
        """Run one of the GX10's local Ollama models (`gemma4:26b`, `qwen3.6:35b`, `nemotron3:33b`, `nemotron-3-super:latest`).
        Vision-capable models receive the live camera JPEG, the 3-keyframe BEV trajectory telemetry, AND the
        bounding-box/depth table (`active_tracks_summary`) to double-check and calibrate target distances."""
        resolved_model = "gemma4:26b" if model_name == "ollama" else model_name
        is_vision = resolved_model in self.GX10_VISION_OLLAMA_MODELS
        tracks_table = (scenario or {}).get("active_tracks_summary") or []
        tracks_json = json.dumps(tracks_table[:6])
        prompt = (
            "You are the GX10 Spatial Perception, Depth Verifier & Haptic Navigation Copilot for a blind/low-vision wearable.\n"
            f"3-second BEV trajectory telemetry: {state_text}\n"
            f"Tracked targets with fast monocular depth estimates (foot_y_pct=1.0 is bottom of frame/closest, 0.0 is top/farthest): {tracks_json}\n"
            "Double-check the target distances based on camera perspective, ground contact (foot_y_pct), and box height (box_h_pct), "
            "then return JSON with:\n"
            "- hazard_choice (approaching_center, crossing_left_to_center, crossing_right_to_center, left_obstacle, right_obstacle, unrealistic_or_clear)\n"
            "- confidence (0.0 to 1.0)\n"
            "- scene_scale_factor (0.5 to 1.5 multiplier to correct fast depth estimates for camera elevation/perspective)\n"
            "- horizon_ratio (0.06 to 0.45 vertical horizon position in frame)\n"
            "- recommended_radar_range_m (6.0 for indoor close, 8.0-12.0 for wide/outdoor scenes so all targets fit inside the radar)\n"
            "- track_depths (array of {track_id, verified_depth_m, verified_x_m} with corrected metric distance in meters for each tracked target)\n"
            "- scene_obstacles (array of {name, zone: LEFT|CENTER|RIGHT, distance_m})\n"
            "- depth_reasoning (1 concise sentence explaining your depth double-check)\n"
            "- guidance (concise 1-sentence navigation cue)"
        )
        msg = {"role": "user", "content": prompt}
        if is_vision and jpeg_bytes:
            msg["images"] = [base64.b64encode(jpeg_bytes).decode("ascii")]

        payload = {
            "model": resolved_model,
            "messages": [msg],
            "format": self.OLLAMA_SCHEMA,
            "think": False,
            "stream": False,
            "keep_alive": "30m",
            "options": {"temperature": 0, "num_predict": 240},
        }
        req = urllib.request.Request(
            f"{self.ollama_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.ollama_timeout_s) as resp:
            body = json.load(resp)
        content = (body.get("message") or {}).get("content") or "{}"
        parsed = json.loads(content)
        choice = parsed.get("hazard_choice", "approaching_center")
        if choice not in self.TREND_TO_3ZONES:
            choice = "approaching_center"
        conf = max(0.5, min(0.99, float(parsed.get("confidence", 0.92))))
        z3 = self.TREND_TO_3ZONES.get(choice, [0, 0, 0])
        motor_probs = [round(conf if v else (1.0 - conf) * 0.25, 3) for v in z3]
        probs = {k: (round(conf, 3) if k == choice else round((1.0 - conf) / 5.0, 4)) for k in self.TREND_TO_3ZONES}
        vlm_obstacles = parsed.get("scene_obstacles") or []
        if vlm_obstacles and callable(self.on_vlm_obstacles):
            try:
                self.on_vlm_obstacles(vlm_obstacles)
            except Exception:
                pass

        # Apply asynchronous VLM depth double-check & BEV map calibration
        track_depths = parsed.get("track_depths") or []
        scale_factor = parsed.get("scene_scale_factor")
        horizon_ratio = parsed.get("horizon_ratio")
        rec_range_m = parsed.get("recommended_radar_range_m")
        depth_reasoning = str(parsed.get("depth_reasoning") or "").strip()
        depth_cal = {
            "model": resolved_model,
            "model_name": resolved_model,
            "scene_scale_factor": scale_factor,
            "horizon_ratio": horizon_ratio,
            "recommended_radar_range_m": rec_range_m,
            "track_depths": track_depths,
            "depth_reasoning": depth_reasoning,
            "updated_at": time.monotonic(),
        }
        with self._lock:
            self._latest_depth_verification = depth_cal
        if callable(self.on_vlm_depth_calibration):
            try:
                self.on_vlm_depth_calibration(depth_cal)
            except Exception:
                pass

        return {
            "backend": resolved_model,
            "model": resolved_model,
            "choice": choice,
            "confidence": round(conf, 3),
            "probabilities": probs,
            "motor_probabilities_3": motor_probs,
            "vlm_obstacles": vlm_obstacles,
            "vlm_guidance": str(parsed.get("guidance") or "").strip(),
            "vlm_depth_verification": depth_cal,
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

    def evaluate_sync(self, scenario, jpeg_bytes=None):
        """Evaluate the 3-keyframe scenario synchronously and return the structured decision JSON."""
        t0 = time.perf_counter()
        state_text = self.build_keyframe_state_text(scenario)
        with self._lock:
            req_backend = self.requested_backend
            if jpeg_bytes is None:
                jpeg_bytes = self._latest_jpeg

        raw_decision = None
        fallback_note = None
        if req_backend in ("ollama",) + self.GX10_OLLAMA_MODELS:
            try:
                raw_decision = self._run_ollama(req_backend, state_text, jpeg_bytes=jpeg_bytes, scenario=scenario)
            except Exception as exc:
                fallback_note = f"GX10 Ollama ({req_backend}) unreachable ({type(exc).__name__}), using Laya fallback"
                raw_decision = self._run_laya(state_text)
        elif req_backend == "jev":
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
        vlm_guidance = raw_decision.get("vlm_guidance")
        vlm_obstacles = raw_decision.get("vlm_obstacles") or []
        realistic_detection = bool(choice != "unrealistic_or_clear" and (kin_realistic or bool(vlm_obstacles)))
        zones_3 = list(self.TREND_TO_3ZONES.get(choice, [0, 0, 0]))
        if realistic_detection and (confirmed > 0 or vlm_obstacles) and sum(zones_3) == 0:
            zones_3 = list(scenario.get("recommended_pre_alert_zones_3") or [0, 1, 0])
        elif not realistic_detection or (confirmed == 0 and not vlm_obstacles):
            zones_3 = [0, 0, 0]

        zones_6 = three_to_six_zones(zones_3)
        haptic_pattern = self.TREND_TO_HAPTIC_PATTERN.get(choice, "silent") if realistic_detection else "silent"

        if vlm_guidance:
            predicted_next_3s = f"{vlm_guidance} → motors {zones_3}"
        elif not realistic_detection:
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

        with self._lock:
            latest_depth_ver = raw_decision.get("vlm_depth_verification") or self._latest_depth_verification

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
            "vlm_obstacles": vlm_obstacles,
            "vlm_guidance": vlm_guidance,
            "vlm_depth_verification": latest_depth_ver,
            "depth_reasoning": (latest_depth_ver or {}).get("depth_reasoning", "") if isinstance(latest_depth_ver, dict) else "",
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
        if self.enable_bg_depth_verifier and self._depth_thread is None:
            self._depth_thread = threading.Thread(target=self._depth_verifier_loop, daemon=True)
            self._depth_thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.5)
        if self._depth_thread:
            self._depth_thread.join(timeout=0.5)

    def submit(self, kinematic_scenario, jpeg_bytes=None):
        with self._lock:
            self._latest_input = dict(kinematic_scenario)
            self._latest_depth_scenario = dict(kinematic_scenario)
            if jpeg_bytes is not None:
                self._latest_jpeg = jpeg_bytes

    def latest(self):
        with self._lock:
            return dict(self._latest_decision) if self._latest_decision else None

    def _loop(self):
        while not self._stop.is_set():
            with self._lock:
                scenario = self._latest_input
                jpeg_bytes = self._latest_jpeg
                self._latest_input = None
            if scenario is None:
                self._stop.wait(0.08)
                continue
            try:
                self.evaluate_sync(scenario, jpeg_bytes=jpeg_bytes)
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def _depth_verifier_loop(self):
        """Asynchronous System-2 GX10 Local VLM loop that runs every ~2.5s in the background when
        a fast System-1 backend (like Laya or Kinematic) is selected, double-checking object depths
        and updating the BEV spatial map around the wearer without blocking real-time haptics."""
        while not self._stop.is_set():
            with self._lock:
                req_backend = self.requested_backend
                scenario = getattr(self, "_latest_depth_scenario", None)
                jpeg_bytes = self._latest_jpeg
            # If req_backend is already an Ollama VLM, _loop already runs _run_ollama
            if req_backend in ("ollama",) + self.GX10_OLLAMA_MODELS or scenario is None:
                self._stop.wait(1.0)
                continue
            if not (scenario.get("active_tracks_summary") or scenario.get("confirmed_tracks")):
                self._stop.wait(0.8)
                continue
            try:
                state_text = self.build_keyframe_state_text(scenario)
                self._run_ollama(
                    self.depth_verifier_model,
                    state_text,
                    jpeg_bytes=jpeg_bytes,
                    scenario=scenario,
                )
            except Exception:
                pass
            self._stop.wait(2.5)


# Keep alias for backwards compatibility
AsyncVLMForecaster = DecisionForecaster



