"""Loopback-only dashboard. Use an SSH tunnel to inspect a remote GX10."""
import argparse
from collections import deque
import base64
import hashlib
import io
import http.client
import json
import math
import os
from pathlib import Path
import select
import socket
import struct
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image, ImageDraw
from software.vision import (
    DEFAULT_OBSTACLE_CLASSES,
    GX10_DETECTOR_MODELS,
    OBSTACLE_CLASS_PRESETS,
    Pipeline,
    MAX_BYTES,
    decode_jpeg,
    resolve_classes,
    resolve_detector_model,
    zone_states,
)
from software.predictor import (
    DecisionForecaster,
    GX10_OLLAMA_MODELS,
    MonocularDepthEstimator,
    RIFEInterpolator,
    TrajectoryPredictor,
    six_to_three_zones,
)
from software.haptics import Controller as HapticController, Settings as HapticSettings, handler_for as haptic_handler
from .geometry import calibration, project

STATIC = Path(__file__).parent

ESP32_FRAMESIZE_MAP = {
    '96x96': (0, '96x96'),
    'qqvga': (1, '160x120'),
    '160x120': (1, '160x120'),
    'qcif': (2, '176x144'),
    '176x144': (2, '176x144'),
    'hqvga': (3, '240x176'),
    '240x176': (3, '240x176'),
    '240x240': (4, '240x240'),
    'qvga': (5, '320x240'),
    '320x240': (5, '320x240'),
    'cif': (6, '400x296'),
    '400x296': (6, '400x296'),
    'hvga': (7, '480x320'),
    '480x320': (7, '480x320'),
    'vga': (8, '640x480'),
    '640x480': (8, '640x480'),
    'svga': (9, '800x600'),
    '800x600': (9, '800x600'),
}
VALID_STREAM_MODES = ('auto', 'mjpeg', 'ws', 'snapshot')


def normalize_camera_resolution(spec='320x240'):
    key = str(spec or '320x240').strip().lower()
    if key not in ESP32_FRAMESIZE_MAP:
        raise ValueError(f'Unsupported camera resolution: {spec}')
    return ESP32_FRAMESIZE_MAP[key]


class StreamBroadcaster:
    """Thread-safe fan-out broadcaster for live MJPEG stream clients with adaptive RIFE frame pacing."""
    def __init__(self):
        self._lock = threading.Lock()
        self._subscribers = []
        self._latest_jpeg = None
        self._cond = threading.Condition(self._lock)
        self._pacer_seq = 0

    def subscribe(self):
        q = deque(maxlen=8)
        with self._lock:
            self._subscribers.append(q)
            if self._latest_jpeg is not None:
                q.append(self._latest_jpeg)
        return q

    def unsubscribe(self, q):
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def broadcast(self, jpeg_bytes):
        with self._lock:
            self._latest_jpeg = jpeg_bytes
            for q in list(self._subscribers):
                q.append(jpeg_bytes)
            self._cond.notify_all()

    def broadcast_paced(self, frames_out, target_interval_s=0.20):
        """Immediately broadcast first frame and evenly pace remaining intermediate/target frames across gap."""
        if not frames_out:
            return
        with self._lock:
            self._pacer_seq += 1
            seq = self._pacer_seq
        # Always broadcast the first frame immediately for zero-blocking responsiveness
        self.broadcast(frames_out[0][0])
        if len(frames_out) <= 1:
            return
        step_s = max(0.020, min(0.140, target_interval_s / float(len(frames_out))))
        remaining = [f[0] for f in frames_out[1:]]

        def _pace():
            for jpeg_bytes in remaining:
                time.sleep(step_s)
                with self._lock:
                    if self._pacer_seq != seq:
                        return
                self.broadcast(jpeg_bytes)

        threading.Thread(target=_pace, daemon=True).start()


class State:
    def __init__(self, mode, output, max_age=2.5, config=None, rotate=0,
                 rife_factor=0, device='auto', decision_backend='laya',
                 jev_key_file='private/jev_api_key.txt', start_decision_worker=False,
                 memory_ttl_s=12.0, depth_mode='auto', classes_spec='indoor_nav',
                 model_spec='auto', ollama_url='http://127.0.0.1:11434',
                 camera_resolution='320x240', camera_quality=14, stream_mode='auto',
                 haptic_settings=None):
        self.lock = threading.Lock()
        self.mode, self.output, self.max_age = mode, Path(output), max_age
        self.rotate = rotate
        self.frame = None
        self.started = None
        self.error = None
        self.paused = False
        self.config = config
        self.generation = 0
        self.observation_sequence = 0
        self.haptics = HapticController(haptic_settings)
        self.classes_spec = classes_spec if classes_spec else 'indoor_nav'
        self.model_spec = str(model_spec or 'auto').strip()
        _, norm_res = normalize_camera_resolution(camera_resolution)
        self.camera_resolution = norm_res
        self.camera_quality = max(4, min(63, int(camera_quality)))
        self.stream_mode = stream_mode if stream_mode in VALID_STREAM_MODES else 'auto'
        self.camera_transport = 'mjpeg_stream' if self.stream_mode in ('auto', 'mjpeg') else (
            'websocket_stream' if self.stream_mode == 'ws' else 'http_keepalive'
        )
        self.camera_config_seq = 0
        self.arrivals = deque(maxlen=30)
        self.stream_arrivals = deque(maxlen=90)
        self.broadcaster = StreamBroadcaster()
        self.predictor = TrajectoryPredictor(memory_ttl_s=memory_ttl_s)
        self.depth_estimator = MonocularDepthEstimator(mode=depth_mode, device=device)
        self.interpolator = RIFEInterpolator(factor=rife_factor, device=device)

        def _on_vlm_depth_cal(cal):
            if not isinstance(cal, dict):
                return
            self.depth_estimator.update_vlm_calibration(
                scale_factor=cal.get('scene_scale_factor'),
                horizon_ratio=cal.get('horizon_ratio'),
                track_depths=cal.get('track_depths'),
                reasoning_note=cal.get('depth_reasoning', ''),
                model_name=cal.get('model_name', ''),
            )
            self.predictor.apply_vlm_depth_verification(
                scale_factor=cal.get('scene_scale_factor'),
                track_depths=cal.get('track_depths'),
                recommended_radar_range_m=cal.get('recommended_radar_range_m'),
                depth_reasoning=cal.get('depth_reasoning', ''),
                model_name=cal.get('model_name', ''),
            )

        self.decision_forecaster = DecisionForecaster(
            backend=decision_backend,
            device=device,
            jev_key_file=jev_key_file,
            ollama_url=ollama_url,
            on_vlm_obstacles=self.predictor.inject_vlm_obstacles,
            on_vlm_depth_calibration=_on_vlm_depth_cal,
            enable_bg_depth_verifier=start_decision_worker,
        )
        if start_decision_worker:
            self.decision_forecaster.start()

    def publish(self, data, result, started, generation, frame_bgr=None):
        with self.lock:
            if self.paused or generation != self.generation:
                return
            config = self.config
            width = result.get('width', 640)
            height = result.get('height', 480)
            raw_objects = []
            for obj in result.get('objects', []):
                xy = project(obj['xyxy'], width, height, config) if config else None
                raw_objects.append({
                    **obj,
                    'ground_xy': xy,
                    'source': 'yolo',
                    'position_kind': 'floor_estimate' if xy else 'bearing_only'
                })
            # Enrich with GX10 Monocular Metric Depth when floor calibration is absent (or add depth_m)
            depth_objects, depth_info = self.depth_estimator.estimate_objects(
                raw_objects, width, height, frame_bgr=frame_bgr, calibration_config=config, jpeg_bytes=data
            )
            enriched_objects, scenario = self.predictor.update(
                depth_objects, width, height, timestamp=started, calibration_config=config
            )

        with self.lock:
            if self.paused or generation != self.generation:
                return
            self.observation_sequence += 1
            scenario['haptic_observation'] = {
                'generation': generation, 'sequence': self.observation_sequence,
                'started': started, 'source_mode': result.get('source_mode', 'unknown'),
                'objects': [{k: obj.get(k) for k in
                             ('current_zone_3', 'depth_m', 'realistic', 'remembered')}
                            for obj in enriched_objects],
            }
        self.decision_forecaster.submit(scenario, jpeg_bytes=data)
        latest_dec = self.decision_forecaster.latest()
        if latest_dec is not None:
            scenario['decision'] = latest_dec

        frames_out, rife_info = self.interpolator.push_and_interpolate_jpegs(data, frame_bgr=frame_bgr)
        now = time.monotonic()
        zones_6 = result.get('zones', [-1] * 6)
        zones_3 = result.get('zones_3') or six_to_three_zones(zones_6)
        with self.lock:
            if self.paused or generation != self.generation:
                return
            if len(self.arrivals) >= 2:
                span_n = min(len(self.arrivals) - 1, 4)
                avg_gap_s = (now - self.arrivals[-span_n]) / span_n
            elif len(self.arrivals) == 1:
                avg_gap_s = now - self.arrivals[-1]
            else:
                avg_gap_s = 0.24
            avg_gap_s = max(0.08, min(1.20, avg_gap_s))
            self.frame = {
                **result,
                'zones_3': zones_3,
                'objects': enriched_objects,
                'remembered_objects': scenario.get('remembered_objects', []),
                'occupancy_cells': scenario.get('occupancy_cells', []),
                'free_space_contour': scenario.get('free_space_contour', []),
                'radar_max_range_m': scenario.get('radar_max_range_m', 7.0),
                'vlm_depth_report': scenario.get('vlm_depth_report'),
                'scenario': scenario,
                'depth': depth_info,
                'predicted_zones_1s': scenario['predicted_zones_1s'],
                'predicted_zones_2s': scenario['predicted_zones_2s'],
                'predicted_zones_3s': scenario['predicted_zones_3s'],
                'predicted_zones_3_2s': scenario['predicted_zones_3_2s'],
                'rife': rife_info,
                'image': base64.b64encode(data).decode('ascii'),
            }
            self.started, self.error = started, None
            self.arrivals.append(now)
            for _ in frames_out:
                self.stream_arrivals.append(now)
        self.broadcaster.broadcast_paced(frames_out, target_interval_s=avg_gap_s)

    def _haptic_locked(self, decision=None):
        decision = decision if decision is not None else self.decision_forecaster.latest()
        frame_fresh = (self.frame is not None and self.started is not None
                       and 0 <= time.monotonic()-self.started < self.max_age
                       and not self.paused and not self.error
                       and self.frame.get('status', 'ok') == 'ok')
        backend_matches = bool(decision and decision.get('requested_backend') == self.decision_forecaster.requested_backend)
        return self.haptics.command(decision, self.generation, enabled=frame_fresh and backend_matches)

    def haptic_command(self):
        with self.lock:
            command = self._haptic_locked()
            # Simulated/replayed signals remain visible as previews, never drive hardware.
            if self.mode != 'live' or command.get('source_mode') != 'live':
                return {**command, 'valid': False, 'status': 'preview_only',
                        'motor_valid_3': [False]*3, 'motor_strength_3': [0]*3, 'ttl_ms': 0}
            return command

    def snapshot(self):
        with self.lock:
            frame = dict(self.frame) if self.frame else None
            age = time.monotonic()-self.started if self.started is not None else None
            valid = bool(frame and age is not None and not self.error and not self.paused and age < self.max_age and frame.get('status','ok') == 'ok')
            status = 'paused' if self.paused else 'error' if self.error else 'waiting' if frame is None or age is None else 'live' if valid else 'stale'
            config = self.config
            depth_active = self.depth_estimator.mode != 'off'
            objects = []
            if valid:
                for obj in frame.get('objects', []):
                    if config:
                        xy = project(obj['xyxy'], frame['width'], frame['height'], config)
                        pos_kind = 'floor_estimate' if xy else 'bearing_only'
                    elif depth_active and obj.get('depth_xy'):
                        xy = obj.get('depth_xy')
                        pos_kind = 'monocular_depth'
                    else:
                        xy = None
                        pos_kind = 'bearing_only'
                    objects.append({
                        **obj,
                        'ground_xy': xy,
                        'source': 'yolo',
                        'position_kind': pos_kind,
                    })
            capture_fps = (
                round((len(self.arrivals)-1)/(self.arrivals[-1]-self.arrivals[0]),2)
                if valid and len(self.arrivals)>1 and self.arrivals[-1]>self.arrivals[0] else 0
            )
            rife_mult = (1 + self.interpolator.factor) if self.interpolator.factor > 0 else 1
            stream_fps = round(capture_fps * rife_mult, 2) if valid and capture_fps > 0 else 0
            scenario = dict(frame.get('scenario')) if valid and frame and frame.get('scenario') else None
            decision = self.decision_forecaster.latest()
            if scenario and decision:
                scenario['decision'] = decision
            rife_info = frame.get('rife') if frame else {'rife_enabled': self.interpolator.factor > 0, 'factor': self.interpolator.factor, 'backend': self.interpolator.backend}
            depth_info = frame.get('depth') if frame else {'enabled': depth_active, 'mode': self.depth_estimator.mode, 'backend': self.depth_estimator._active_backend, 'depth_ms': 0.0}
            zones_6 = frame['zones'] if valid else [-1]*6
            zones_3 = frame.get('zones_3', six_to_three_zones(zones_6)) if valid else [-1]*3
            remembered_objects = scenario.get('remembered_objects', []) if scenario else []
            occupancy_cells = scenario.get('occupancy_cells', []) if scenario else []
            free_space_contour = scenario.get('free_space_contour', []) if scenario else []
            radar_max_range_m = scenario.get('radar_max_range_m', 7.0) if scenario else 7.0
            vlm_depth_report = scenario.get('vlm_depth_report') if scenario else None
            map_mode = 'floor_estimate' if config else ('monocular_depth' if depth_active else 'bearing_only')
            return {'mode': self.mode, 'status': status, 'valid': valid,
                    'error': self.error, 'age_ms': round(age*1000) if age is not None else None,
                    'max_age_ms': round(self.max_age*1000), 'frame': frame,
                    'capture_fps': capture_fps,
                    'stream_fps': stream_fps,
                    'camera_resolution': self.camera_resolution,
                    'camera_quality': self.camera_quality,
                    'stream_mode': self.stream_mode,
                    'camera_transport': self.camera_transport,
                    'rife': rife_info,
                    'depth': depth_info,
                    'depth_mode': self.depth_estimator.mode,
                    'memory_ttl_s': self.predictor.memory_ttl_s,
                    'classes_spec': self.classes_spec,
                    'model_spec': self.model_spec,
                    'scenario': scenario,
                    'decision': decision,
                    'haptics': {**self._haptic_locked(decision), 'preview_only': self.mode != 'live'},
                    'decision_backend': self.decision_forecaster.requested_backend,
                    'predicted_zones_1s': scenario['predicted_zones_1s'] if scenario else [-1]*6,
                    'predicted_zones_2s': scenario['predicted_zones_2s'] if scenario else [-1]*6,
                    'predicted_zones_3s': scenario['predicted_zones_3s'] if scenario else [-1]*6,
                    'predicted_zones_3_2s': (decision['recommended_pre_alert_zones_3'] if decision else scenario.get('predicted_zones_3_2s', [-1]*3)) if scenario else [-1]*3,
                    'objects': objects,
                    'remembered_objects': remembered_objects,
                    'occupancy_cells': occupancy_cells,
                    'free_space_contour': free_space_contour,
                    'radar_max_range_m': radar_max_range_m,
                    'vlm_depth_report': vlm_depth_report,
                    'zones': zones_6, 'zones_3': zones_3,
                    'calibration': config, 'map_mode': map_mode,
                    'rotate': self.rotate}

    def control(self, paused):
        with self.lock:
            self.paused = paused
            self.generation += 1
            self.arrivals.clear()
            self.stream_arrivals.clear()
            # A resume must wait for a newly captured frame.
            self.started = None
        self.predictor.reset()
        self.interpolator.reset()

    def set_rotate(self, rotate):
        with self.lock:
            if rotate not in (0, 90, 180, 270):
                raise ValueError('Rotation must be 0, 90, 180, or 270')
            self.rotate = rotate
            self.generation += 1
            self.arrivals.clear()
            self.stream_arrivals.clear()
            self.started = None
        self.predictor.reset()
        self.interpolator.reset()

    def set_mode(self, mode):
        with self.lock:
            if mode not in ('live', 'simulation', 'replay'):
                raise ValueError('Mode must be live, simulation, or replay')
            self.mode = mode
            self.rotate = 270 if mode == 'live' else 0
            self.generation += 1
            self.frame = None
            self.arrivals.clear()
            self.stream_arrivals.clear()
            self.started = None
            self.error = None
        with self.broadcaster._lock:
            self.broadcaster._pacer_seq += 1
            self.broadcaster._latest_jpeg = None
            for q in self.broadcaster._subscribers:
                q.clear()
        self.predictor.reset()
        self.interpolator.reset()

    def set_rife(self, factor):
        if type(factor) is not int or not 0 <= factor <= 8:
            raise ValueError('rife factor must be an integer between 0 and 8')
        with self.lock:
            self.interpolator.factor = factor
            self.interpolator.stats['factor'] = factor
        self.interpolator.reset()

    def set_memory_ttl(self, memory_ttl_s):
        if type(memory_ttl_s) not in (int, float) or not math.isfinite(memory_ttl_s) or not 0 <= float(memory_ttl_s) <= 60:
            raise ValueError('memory_ttl_s must be a number between 0 and 60')
        self.predictor.set_memory_ttl(float(memory_ttl_s))

    def set_depth_mode(self, mode):
        self.depth_estimator.set_mode(mode)

    def set_classes(self, classes_spec):
        resolved = resolve_classes(classes_spec)
        if not resolved:
            raise ValueError('At least one obstacle class is required')
        with self.lock:
            self.classes_spec = str(classes_spec).strip()

    def set_model(self, model_spec):
        spec = str(model_spec or '').strip()
        if not spec:
            raise ValueError('model must be a non-empty detector model name or auto')
        with self.lock:
            self.model_spec = spec

    def set_camera_config(self, resolution=None, quality=None, stream_mode=None):
        with self.lock:
            if resolution is not None:
                _, norm_res = normalize_camera_resolution(resolution)
                self.camera_resolution = norm_res
            if quality is not None:
                if type(quality) is bool or type(quality) is not int or not 4 <= quality <= 63:
                    raise ValueError('quality must be an integer between 4 and 63')
                self.camera_quality = quality
            if stream_mode is not None:
                if stream_mode not in VALID_STREAM_MODES:
                    raise ValueError(f'stream_mode must be one of {VALID_STREAM_MODES}')
                self.stream_mode = stream_mode
            self.camera_config_seq += 1
            self.arrivals.clear()
            self.stream_arrivals.clear()
            return {
                'camera_resolution': self.camera_resolution,
                'camera_quality': self.camera_quality,
                'stream_mode': self.stream_mode,
            }

    def set_decision_backend(self, backend):
        self.decision_forecaster.set_backend(backend)
        self.decision_forecaster.start()
        with self.lock:
            scen = dict(self.frame.get('scenario')) if self.frame and self.frame.get('scenario') else None
        if scen is not None:
            try:
                dec = self.decision_forecaster.evaluate_sync(scen)
                with self.lock:
                    if self.frame and self.frame.get('scenario'):
                        self.frame['scenario']['decision'] = dec
            except Exception:
                pass


def demo_frame(index):
    im = Image.new('RGB', (640,480), '#d8e0e4'); draw = ImageDraw.Draw(im)
    draw.rectangle((0,240,640,480), fill='#b7c2c5')
    for line_x in range(-640,1280,120): draw.line((320,240,line_x,480),fill='#a0adb3',width=2)
    # Smooth sinusoidal human running pace across sensor field
    phase = ((index % 60) / 60.0) * 2 * math.pi
    x = int(280 + 200 * math.sin(phase))
    draw.ellipse((x+20,125,x+60,165),fill='#3f5d70')
    draw.rounded_rectangle((x,168,x+80,340),radius=15,fill='#3f5d70')
    draw.rectangle((x+5,330,x+28,410),fill='#3f5d70');draw.rectangle((x+52,330,x+75,410),fill='#3f5d70')

    objects = [{'class': 'person', 'confidence': .94, 'xyxy': [x, 125, x+80, 410]}]
    box_spans = [(x, x+80)]

    # Stationary chair on left-peripheral that periodically slips out of FOV to demonstrate 12s Spatial Memory
    if (index % 90) < 62:
        cx1, cy1, cx2, cy2 = 52, 235, 134, 385
        draw.rounded_rectangle((cx1, cy1, cx2, cy2), radius=8, fill='#5b4b3a')
        objects.append({'class': 'chair', 'confidence': .89, 'xyxy': [cx1, cy1, cx2, cy2]})
        box_spans.append((cx1, cx2))

    # Stationary dining table / backpack on right side that alternates visibility
    if 15 <= (index % 90) < 78:
        tx1, ty1, tx2, ty2 = 490, 225, 605, 368
        draw.rounded_rectangle((tx1, ty1, tx2, ty2), radius=6, fill='#4a5568')
        objects.append({'class': 'dining table', 'confidence': .87, 'xyxy': [tx1, ty1, tx2, ty2]})
        box_spans.append((tx1, tx2))

    draw.text((20,20),'SIMULATION / NOT A CAMERA CAPTURE',fill='#273947')
    data=io.BytesIO();im.save(data,format='JPEG')
    return data.getvalue(), {'width':640,'height':480,'objects':objects,
                             'zones':zone_states(box_spans,640,[i/6 for i in range(7)])}


def _set_tcp_nodelay(sock):
    if sock is None:
        return
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except OSError:
        pass


def configure_esp32_camera(url, resolution='320x240', quality=14, timeout=0.75):
    """Configure ESP32 OV2640/OV3660 sensor to low-res QVGA (320x240) and fast JPEG quality via /control."""
    framesize_idx, norm_res = normalize_camera_resolution(resolution)
    quality_val = max(4, min(63, int(quality)))
    parsed = urllib.parse.urlsplit(url)
    host = parsed.hostname
    if not host:
        return {'ok': False, 'framesize': framesize_idx, 'resolution': norm_res, 'quality': quality_val}
    scheme = 'https' if parsed.scheme in ('https', 'wss') else 'http'
    port = parsed.port or (443 if scheme == 'https' else 80)
    candidate_ports = [port]
    if port == 81:
        candidate_ports = [80, 81]
    applied = False
    for p in candidate_ports:
        base = f'{scheme}://{host}:{p}'
        try:
            for var, val in (('framesize', framesize_idx), ('quality', quality_val)):
                req = urllib.request.Request(
                    f'{base}/control?var={var}&val={val}',
                    headers={'Cache-Control': 'no-cache', 'Connection': 'close'},
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    if resp.status == 200:
                        applied = True
            if applied:
                break
        except Exception:
            continue
    return {'ok': applied, 'framesize': framesize_idx, 'resolution': norm_res, 'quality': quality_val}


class SnapshotCamera:
    """Reuse the camera HTTP connection; reconnect cleanly when closed without overwhelming ESP32 LWIP."""
    is_persistent_stream = False
    transport = 'http_keepalive'

    def __init__(self, url, timeout):
        self.url_str = url
        self.url = urllib.parse.urlsplit(url)
        self.timeout = min(float(timeout), 2.5) if timeout > 0 else timeout
        self.cls = http.client.HTTPSConnection if self.url.scheme == 'https' else http.client.HTTPConnection
        self.connection = self.cls(self.url.hostname, self.url.port, timeout=self.timeout)

    def _request_once(self):
        if self.connection.sock is not None:
            _set_tcp_nodelay(self.connection.sock)
            try:
                self.connection.sock.settimeout(min(self.timeout, 1.8))
            except OSError:
                pass
        query = self.url.query + ('&' if self.url.query else '') + '_cb=' + str(time.time_ns())
        path = urllib.parse.urlunsplit(('', '', self.url.path or '/', query, ''))
        self.connection.request('GET', path, headers={'Cache-Control': 'no-cache', 'Accept': 'image/jpeg'})
        _set_tcp_nodelay(self.connection.sock)
        response = self.connection.getresponse()
        if response.status != 200:
            response.close()
            raise OSError(f'Camera HTTP {response.status}')
        data = response.read(MAX_BYTES + 1)
        will_close = getattr(response, 'will_close', False)
        response.close()
        if will_close:
            self.close()
        if len(data) > MAX_BYTES:
            raise ValueError('Camera JPEG exceeds the size limit')
        return data

    def read(self):
        had_socket = (self.connection.sock is not None)
        try:
            return self._request_once()
        except (http.client.RemoteDisconnected, ConnectionResetError, BrokenPipeError,
                http.client.BadStatusLine, http.client.CannotSendRequest,
                http.client.ResponseNotReady) as exc:
            self.close()
            if had_socket:
                self.connection = self.cls(self.url.hostname, self.url.port, timeout=self.timeout)
                return self._request_once()
            raise
        except Exception:
            self.close()
            self.connection = self.cls(self.url.hostname, self.url.port, timeout=self.timeout)
            raise

    def read_latest(self):
        return self.read()

    def close(self):
        self.connection.close()


class StreamCamera:
    """Reads continuous multipart MJPEG stream over a single persistent TCP_NODELAY connection."""
    is_persistent_stream = True
    transport = 'mjpeg_stream'

    def __init__(self, url, timeout=3.5):
        self.url_str = url
        self.url = urllib.parse.urlsplit(url)
        self.timeout = timeout
        self.response = None
        self.sock = None
        self.buffer = bytearray()
        self._connect()

    def _connect(self):
        self.close()
        req = urllib.request.Request(
            self.url_str,
            headers={'Cache-Control': 'no-cache', 'Accept': 'multipart/x-mixed-replace, image/jpeg, */*'}
        )
        self.response = urllib.request.urlopen(req, timeout=self.timeout)
        sock = None
        try:
            raw = getattr(getattr(self.response, 'fp', None), 'raw', None)
            sock = getattr(raw, '_sock', None)
        except Exception:
            sock = None
        self.sock = sock
        _set_tcp_nodelay(self.sock)
        self.buffer.clear()

    def _read_fn(self):
        read_fn = getattr(self.response, 'read1', None)
        if read_fn is None and hasattr(self.response, 'fp') and hasattr(self.response.fp, 'read1'):
            read_fn = self.response.fp.read1
        if read_fn is None:
            read_fn = self.response.read
        return read_fn

    def _pop_first_frame(self):
        soi = self.buffer.find(b'\xff\xd8')
        if soi != -1:
            eoi = self.buffer.find(b'\xff\xd9', soi + 2)
            if eoi != -1:
                frame = bytes(self.buffer[soi:eoi + 2])
                del self.buffer[:eoi + 2]
                if len(frame) > MAX_BYTES:
                    raise ValueError('Camera JPEG exceeds the size limit')
                return frame
        return None

    def read(self):
        if self.response is None:
            self._connect()
        read_fn = self._read_fn()
        while True:
            frame = self._pop_first_frame()
            if frame is not None:
                return frame
            chunk = read_fn(8192)
            if not chunk:
                self.close()
                raise OSError('Camera stream closed')
            self.buffer.extend(chunk)
            if len(self.buffer) > MAX_BYTES * 2:
                del self.buffer[:-MAX_BYTES]

    def read_latest(self):
        """Return the newest complete JPEG frame, non-blockingly draining any queued burst in the socket."""
        frame = self.read()
        if self.response is None:
            return frame
        if self.sock is not None:
            read_fn = self._read_fn()
            for _ in range(8):
                try:
                    ready, _, _ = select.select([self.sock], [], [], 0.0)
                    if not ready:
                        break
                    chunk = read_fn(16384)
                    if not chunk:
                        break
                    self.buffer.extend(chunk)
                except Exception:
                    break
        while True:
            newer = self._pop_first_frame()
            if newer is None:
                break
            frame = newer
        return frame

    def close(self):
        if self.response:
            try: self.response.close()
            except Exception: pass
            self.response = None
        self.sock = None
        self.buffer.clear()


class WebSocketCamera:
    """Reads continuous binary JPEG frames over a persistent RFC 6455 WebSocket (/ws) with TCP_NODELAY."""
    is_persistent_stream = True
    transport = 'websocket_stream'

    def __init__(self, url, timeout=3.5):
        self.url_str = url
        self.url = urllib.parse.urlsplit(url)
        self.timeout = timeout
        self.sock = None
        self.buffer = bytearray()
        self._connect()

    def _connect(self):
        self.close()
        host = self.url.hostname or '127.0.0.1'
        port = self.url.port or (443 if self.url.scheme in ('wss', 'https') else 80)
        path = self.url.path or '/ws'
        if self.url.query:
            path = f'{path}?{self.url.query}'
        sock = socket.create_connection((host, port), timeout=self.timeout)
        _set_tcp_nodelay(sock)
        key_bytes = base64.b64encode(os.urandom(16)).decode('ascii')
        expected_accept = base64.b64encode(
            hashlib.sha1((key_bytes + '258EAFA5-E914-47DA-95CA-C5AB0DC85B11').encode('ascii')).digest()
        ).decode('ascii')
        handshake = (
            f'GET {path} HTTP/1.1\r\n'
            f'Host: {host}:{port}\r\n'
            f'Upgrade: websocket\r\n'
            f'Connection: Upgrade\r\n'
            f'Sec-WebSocket-Key: {key_bytes}\r\n'
            f'Sec-WebSocket-Version: 13\r\n\r\n'
        ).encode('ascii')
        sock.sendall(handshake)
        header_buf = bytearray()
        while b'\r\n\r\n' not in header_buf:
            chunk = sock.recv(4096)
            if not chunk:
                sock.close()
                raise OSError('WebSocket handshake closed by peer')
            header_buf.extend(chunk)
            if len(header_buf) > 16384:
                sock.close()
                raise OSError('WebSocket handshake headers too large')
        sep = header_buf.index(b'\r\n\r\n')
        header_text = header_buf[:sep].decode('latin1', errors='replace')
        status_line = header_text.splitlines()[0] if header_text else ''
        if '101' not in status_line:
            sock.close()
            raise OSError(f'WebSocket upgrade rejected: {status_line}')
        if expected_accept not in header_text:
            sock.close()
            raise OSError('WebSocket Sec-WebSocket-Accept mismatch')
        self.sock = sock
        self.buffer = bytearray(header_buf[sep + 4:])

    def _ensure_bytes(self, n):
        while len(self.buffer) < n:
            chunk = self.sock.recv(max(8192, n - len(self.buffer)))
            if not chunk:
                self.close()
                raise OSError('WebSocket stream closed')
            self.buffer.extend(chunk)

    def _read_ws_frame(self):
        if self.sock is None:
            self._connect()
        fragments = bytearray()
        while True:
            self._ensure_bytes(2)
            b0, b1 = self.buffer[0], self.buffer[1]
            fin = (b0 & 0x80) != 0
            opcode = b0 & 0x0F
            masked = (b1 & 0x80) != 0
            payload_len = b1 & 0x7F
            offset = 2
            if payload_len == 126:
                self._ensure_bytes(4)
                payload_len = struct.unpack('!H', self.buffer[2:4])[0]
                offset = 4
            elif payload_len == 127:
                self._ensure_bytes(10)
                payload_len = struct.unpack('!Q', self.buffer[2:10])[0]
                offset = 10
            if payload_len > MAX_BYTES:
                self.close()
                raise ValueError('WebSocket frame exceeds size limit')
            mask_key = None
            if masked:
                self._ensure_bytes(offset + 4)
                mask_key = bytes(self.buffer[offset:offset + 4])
                offset += 4
            self._ensure_bytes(offset + payload_len)
            payload = bytearray(self.buffer[offset:offset + payload_len])
            del self.buffer[:offset + payload_len]
            if mask_key:
                for i in range(len(payload)):
                    payload[i] ^= mask_key[i & 3]
            if opcode == 0x8:  # Close
                self.close()
                raise OSError('WebSocket closed by camera')
            if opcode == 0x9:  # Ping -> send masked Pong
                try:
                    pong_mask = os.urandom(4)
                    masked_pong = bytes(payload[i] ^ pong_mask[i & 3] for i in range(min(len(payload), 125)))
                    self.sock.sendall(bytes([0x8A, 0x80 | len(masked_pong)]) + pong_mask + masked_pong)
                except OSError:
                    pass
                continue
            if opcode == 0xA:  # Pong
                continue
            fragments.extend(payload)
            if fin:
                return bytes(fragments)

    def read(self):
        while True:
            data = self._read_ws_frame()
            soi = data.find(b'\xff\xd8')
            eoi = data.rfind(b'\xff\xd9')
            if soi != -1 and eoi != -1 and eoi > soi:
                return data[soi:eoi + 2]

    def read_latest(self):
        frame = self.read()
        if self.sock is None:
            return frame
        for _ in range(6):
            try:
                ready, _, _ = select.select([self.sock], [], [], 0.0)
                if not ready and len(self.buffer) < 2:
                    break
                if ready:
                    chunk = self.sock.recv(16384)
                    if not chunk:
                        break
                    self.buffer.extend(chunk)
                if len(self.buffer) >= 2:
                    b1 = self.buffer[1]
                    plen = b1 & 0x7F
                    hdr = 2 + (2 if plen == 126 else 8 if plen == 127 else 0) + (4 if (b1 & 0x80) else 0)
                    if len(self.buffer) < hdr:
                        break
                    if plen == 126:
                        plen = struct.unpack('!H', self.buffer[2:4])[0]
                    elif plen == 127:
                        plen = struct.unpack('!Q', self.buffer[2:10])[0]
                    if len(self.buffer) < hdr + plen:
                        break
                    cand = self.read()
                    if cand:
                        frame = cand
            except Exception:
                break
        return frame

    def close(self):
        if self.sock:
            try: self.sock.close()
            except Exception: pass
            self.sock = None
        self.buffer.clear()


def _candidate_stream_urls(url):
    parsed = urllib.parse.urlsplit(url)
    path = parsed.path or '/'
    scheme = 'https' if parsed.scheme in ('https', 'wss') else 'http'
    host = parsed.hostname or '127.0.0.1'
    port = parsed.port
    candidates = []
    if '/stream' in path or 'mjpeg' in url.lower() or 'mjpg' in url.lower():
        candidates.append(url)
        if port in (None, 80) and path == '/stream':
            candidates.append(f'{scheme}://{host}:81/stream')
    elif path.endswith('/capture') or path == '/':
        base_netloc = f'{host}:{port}' if port else host
        candidates.append(urllib.parse.urlunsplit((scheme, base_netloc, '/stream', parsed.query, '')))
        if port in (None, 80):
            candidates.append(urllib.parse.urlunsplit((scheme, f'{host}:81', '/stream', parsed.query, '')))
    else:
        candidates.append(url)
    return candidates


def _candidate_ws_urls(url):
    parsed = urllib.parse.urlsplit(url)
    ws_scheme = 'wss' if parsed.scheme in ('https', 'wss') else 'ws'
    host = parsed.hostname or '127.0.0.1'
    port = parsed.port
    base_netloc = f'{host}:{port}' if port else host
    path = parsed.path if parsed.path and parsed.path.endswith('/ws') else '/ws'
    candidates = [urllib.parse.urlunsplit((ws_scheme, base_netloc, path, parsed.query, ''))]
    if port in (None, 80):
        candidates.append(urllib.parse.urlunsplit((ws_scheme, f'{host}:81', '/ws', parsed.query, '')))
    return candidates


def create_camera(url, timeout, is_stream=None, resolution='320x240', quality=14, configure_sensor=False):
    """Instantiate camera reader: StreamCamera / WebSocketCamera for persistent streams, SnapshotCamera fallback."""
    if configure_sensor:
        configure_esp32_camera(url, resolution=resolution, quality=quality, timeout=min(float(timeout), 0.75))

    parsed = urllib.parse.urlsplit(url)
    url_lower = url.lower()
    is_ws_url = parsed.scheme in ('ws', 'wss') or parsed.path.endswith('/ws')
    is_mjpeg_url = '/stream' in parsed.path or 'mjpeg' in url_lower or 'mjpg' in url_lower

    if is_stream == 'snapshot' or is_stream is False:
        return SnapshotCamera(url, timeout)

    if is_stream == 'ws' or (is_stream in (None, True, 'auto') and is_ws_url):
        for ws_url in _candidate_ws_urls(url):
            try:
                return WebSocketCamera(ws_url, timeout)
            except Exception:
                continue
        if is_ws_url:
            fallback_http = urllib.parse.urlunsplit((
                'https' if parsed.scheme == 'wss' else 'http',
                parsed.netloc,
                '/capture',
                parsed.query,
                '',
            ))
            return SnapshotCamera(fallback_http, timeout)

    if is_stream in (True, 'auto', 'mjpeg') or (is_stream is None and is_mjpeg_url):
        for stream_url in _candidate_stream_urls(url):
            try:
                return StreamCamera(stream_url, timeout)
            except Exception:
                continue
        if is_stream == 'auto':
            for ws_url in _candidate_ws_urls(url):
                try:
                    return WebSocketCamera(ws_url, timeout)
                except Exception:
                    continue
        fallback_url = url
        if parsed.path.endswith('/stream'):
            fallback_url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, '/capture', parsed.query, ''))
        return SnapshotCamera(fallback_url, timeout)

    return SnapshotCamera(url, timeout)


class LatestFrameBuffer:
    """Thread-safe single-item drop-oldest buffer. Guarantees inference always gets the freshest frame."""
    def __init__(self):
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._frame = None
        self._timestamp = None
        self._generation = 0
        self._closed = False

    def put(self, item, timestamp=None):
        with self._cond:
            if self._closed:
                return
            self._frame = item
            self._timestamp = timestamp if timestamp is not None else time.monotonic()
            self._generation += 1
            self._cond.notify_all()

    def get(self, last_generation=None, timeout=0.5):
        with self._cond:
            if last_generation is not None and self._generation == last_generation:
                if not self._cond.wait(timeout):
                    return None, None, self._generation
            return self._frame, self._timestamp, self._generation

    def clear(self):
        with self._cond:
            self._frame = None
            self._timestamp = None
            self._generation += 1
            self._cond.notify_all()

    def close(self):
        with self._cond:
            self._closed = True
            self._cond.notify_all()


def prepare_frame_and_rotate(raw_bytes, rotate):
    """Fast in-memory decode and rotation. Returns (rotated_jpeg_bytes, image_input, width, height)."""
    try:
        import cv2
        import numpy as np
        nparr = np.frombuffer(raw_bytes, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr is not None:
            if not rotate:
                h, w = bgr.shape[:2]
                return raw_bytes, bgr, w, h
            if rotate == 90:
                bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
            elif rotate == 180:
                bgr = cv2.rotate(bgr, cv2.ROTATE_180)
            elif rotate == 270:
                bgr = cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
            h, w = bgr.shape[:2]
            _, enc = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return enc.tobytes(), bgr, w, h
    except (ImportError, Exception):
        pass
    if not rotate:
        return raw_bytes, None, None, None
    # Pillow fallback
    image = decode_jpeg(raw_bytes).rotate(-rotate, expand=True)
    output = io.BytesIO()
    image.save(output, format='JPEG', quality=85)
    return output.getvalue(), image, image.width, image.height


def run_source(state, args, stop):
    pipeline = None
    camera = None
    producer_thread = None
    buffer = LatestFrameBuffer()
    try:
        if args.camera_url:
            with state.lock:
                init_res = state.camera_resolution
                init_qual = state.camera_quality
                init_stream_mode = state.stream_mode
                applied_cfg_seq = state.camera_config_seq
            effective_stream = (
                'snapshot' if getattr(args, 'no_stream', False)
                else (True if getattr(args, 'stream', False) else init_stream_mode)
            )
            camera = create_camera(
                args.camera_url,
                args.timeout,
                is_stream=effective_stream,
                resolution=init_res,
                quality=init_qual,
                configure_sensor=True,
            )
            with state.lock:
                state.camera_transport = getattr(camera, 'transport', 'mjpeg_stream')
            pipeline = Pipeline(
                args.output / 'yolo',
                detect=True,
                model=args.model,
                device=args.device,
                classes=getattr(args, 'classes', 'indoor_nav'),
                max_result_age_ms=args.max_age * 1000,
                save_disk=getattr(args, 'save_artifacts', False),
                predict_horizon=False
            )

            # Dedicated producer thread reads camera continuously without blocking YOLO
            def producer_loop():
                nonlocal camera, applied_cfg_seq
                consecutive_errors = 0
                sensor_configured = (getattr(camera, 'transport', '') != 'http_keepalive')
                while not stop.is_set():
                    with state.lock:
                        paused, current_mode = state.paused, state.mode
                        cfg_seq = state.camera_config_seq
                        target_res = state.camera_resolution
                        target_qual = state.camera_quality
                        target_stream_mode = state.stream_mode
                    if paused or current_mode != 'live':
                        if camera: camera.close()
                        stop.wait(0.1)
                        continue
                    if cfg_seq != applied_cfg_seq:
                        applied_cfg_seq = cfg_seq
                        sensor_configured = False
                        try:
                            if camera:
                                camera.close()
                            camera = create_camera(
                                args.camera_url,
                                args.timeout,
                                is_stream=target_stream_mode,
                                resolution=target_res,
                                quality=target_qual,
                                configure_sensor=True,
                            )
                            sensor_configured = True
                            with state.lock:
                                state.camera_transport = getattr(camera, 'transport', 'mjpeg_stream')
                        except Exception:
                            pass
                    started = time.monotonic()
                    try:
                        if not sensor_configured:
                            # ESP32 may have powered on after server start; configure 320x240 & try stream
                            try:
                                new_cam = create_camera(
                                    args.camera_url,
                                    args.timeout,
                                    is_stream=target_stream_mode,
                                    resolution=target_res,
                                    quality=target_qual,
                                    configure_sensor=True,
                                )
                                if camera:
                                    camera.close()
                                camera = new_cam
                                sensor_configured = True
                            except Exception:
                                pass
                        read_fn = getattr(camera, 'read_latest', camera.read)
                        raw = read_fn()
                        fetched = time.monotonic()
                        buffer.put((raw, started, fetched))
                        consecutive_errors = 0
                        with state.lock:
                            state.error = None
                            state.camera_transport = getattr(camera, 'transport', state.camera_transport)
                        if not getattr(camera, 'is_persistent_stream', False):
                            stop.wait(0.015)
                    except Exception as exc:
                        consecutive_errors += 1
                        if consecutive_errors >= 2:
                            sensor_configured = False
                        with state.lock:
                            if consecutive_errors >= 8 or (state.frame is None and consecutive_errors >= 3):
                                state.error = str(exc)
                        stop.wait(0.25 if consecutive_errors < 4 else 0.50)
                if camera:
                    camera.close()

            producer_thread = threading.Thread(target=producer_loop, daemon=True)
            producer_thread.start()

        replay = []
        candidate_replay_dirs = [
            args.replay,
            Path('artifacts/camera-evaluation-20261010/upright-live'),
            Path(__file__).resolve().parents[2] / 'artifacts/camera-evaluation-20261010/upright-live',
            Path(__file__).resolve().parents[4] / 'artifacts/camera-evaluation-20261010/upright-live',
            Path('/home/asus/RAGging-Canes/artifacts/camera-evaluation-20261010/upright-live'),
        ]
        replay_dir = None
        for cand in candidate_replay_dirs:
            if cand and (cand / 'captures.jsonl').exists():
                replay_dir = cand
                break
        if replay_dir:
            try:
                captures = [json.loads(line) for line in (replay_dir / 'captures.jsonl').read_text().splitlines()]
                results = [json.loads(line) for line in (replay_dir / 'results.jsonl').read_text().splitlines()]
                by_id = {r['frame']: r for r in results if r['backend'] == 'yolo' and r['repeat'] == 1 and r.get('valid')}
                for c in captures:
                    if c.get('valid') and c['frame'] in by_id:
                        path = (replay_dir / c['file']).resolve()
                        if path.is_file():
                            data = path.read_bytes()
                            image = decode_jpeg(data)
                            replay.append((data, {**by_id[c['frame']], 'width': image.width, 'height': image.height,
                                                  'source_frame': c['file']}))
            except Exception:
                pass
        if args.replay and not replay:
            raise ValueError('No paired YOLO frames in this replay directory.')

        index = 0
        last_gen = 0
        while not stop.is_set():
            with state.lock:
                paused, generation, rotate, current_mode, classes_spec, model_spec = (
                    state.paused, state.generation, state.rotate, state.mode, state.classes_spec, state.model_spec
                )
            if paused:
                buffer.clear()
                stop.wait(0.1)
                continue

            try:
                frame_bgr = None
                if current_mode == 'live' and args.camera_url:
                    if pipeline is not None and getattr(pipeline, 'model_spec', None) != model_spec:
                        try:
                            pipeline.set_model(model_spec)
                        except Exception:
                            pass
                    if pipeline is not None and getattr(pipeline, 'classes_spec', None) != classes_spec:
                        try:
                            pipeline.set_classes(classes_spec)
                        except Exception:
                            pass
                    item, _, gen = buffer.get(last_gen, timeout=0.1)
                    if item is None or gen == last_gen:
                        continue
                    last_gen = gen
                    raw, started, fetched = item
                    rotated_jpeg, image_input, w, h = prepare_frame_and_rotate(raw, rotate)
                    if image_input is not None and hasattr(image_input, 'shape'):
                        frame_bgr = image_input
                    result = pipeline.process(
                        rotated_jpeg,
                        str(index + 1),
                        observation_started=started,
                        image_input=image_input
                    )
                    result['capture_ms'] = round((fetched - started) * 1000, 2)
                    result['camera_transport'] = getattr(camera, 'transport', state.camera_transport)
                    raw_to_publish = rotated_jpeg
                elif current_mode == 'replay' and replay:
                    started = time.monotonic()
                    raw_to_publish, result = replay[index % len(replay)]
                    result = dict(result)
                    stop.wait(0.35)
                else:
                    started = time.monotonic()
                    raw_to_publish, result = demo_frame(index)
                    stop.wait(0.04)

                result['source_mode'] = ('live' if current_mode == 'live' and args.camera_url
                                         else 'replay' if current_mode == 'replay' and replay else 'simulation')
                result['frame_id'] = index + 1
                state.publish(raw_to_publish, result, started, generation, frame_bgr=frame_bgr)
                index += 1
            except Exception as exc:
                with state.lock:
                    if generation == state.generation:
                        state.error = str(exc)
                stop.wait(0.2)
    except Exception as exc:
        with state.lock:
            state.error = str(exc)
    finally:
        buffer.close()
        if camera:
            camera.close()
        if producer_thread:
            producer_thread.join(timeout=0.5)


def handler_for(state):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        def log_message(self,*args): pass

        def send(self, status, body, mime='application/json'):
            if isinstance(body,dict): body=json.dumps(body).encode()
            self.send_response(status);self.send_header('Content-Type',mime)
            self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; frame-ancestors 'none'")
            self.end_headers();self.wfile.write(body)

        def allowed(self):
            host=self.headers.get('Host','')
            hostname=urllib.parse.urlsplit('//'+host).hostname
            if hostname not in ('127.0.0.1','localhost','[::1]','::1'):
                self.send(403,{'error':'Use a localhost URL or SSH tunnel.'});return False
            origin=self.headers.get('Origin')
            if origin and origin != 'http://'+host:
                self.send(403,{'error':'Cross-origin requests are disabled.'});return False
            return True

        def do_GET(self):
            if not self.allowed(): return
            path=urllib.parse.urlsplit(self.path).path

            if path == '/stream.mjpg':
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                self.send_header('Pragma', 'no-cache')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.end_headers()
                q = state.broadcaster.subscribe()
                try:
                    while True:
                        with state.broadcaster._lock:
                            if not q:
                                state.broadcaster._cond.wait(timeout=0.2)
                            jpeg = q.popleft() if q else None
                        if jpeg:
                            header = (
                                b'--frame\r\n'
                                b'Content-Type: image/jpeg\r\n'
                                b'Content-Length: ' + str(len(jpeg)).encode() + b'\r\n\r\n'
                            )
                            self.wfile.write(header + jpeg + b'\r\n')
                            self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    state.broadcaster.unsubscribe(q)
                return

            if path=='/api/state':
                snapshot=state.snapshot()
                qs = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
                after = qs.get('after', [None])[0]
                meta_only = qs.get('meta_only', ['0'])[0] in ('1', 'true', 'yes')
                if snapshot['frame'] and (meta_only or after == str(snapshot['frame']['frame_id'])):
                    snapshot['frame'].pop('image',None)
                return self.send(200,snapshot)
            files={'/':('index.html','text/html'),'/app.js':('app.js','text/javascript'),'/style.css':('style.css','text/css')}
            if path in files:
                file,mime=files[path];return self.send(200,(STATIC/file).read_bytes(),mime)
            self.send(404,{'error':'Not found'})

        def do_POST(self):
            if not self.allowed(): return
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0 < size <= 8192: raise ValueError('Invalid request size.')
                payload=json.loads(self.rfile.read(size))
                if not isinstance(payload,dict): raise ValueError('Expected JSON object.')
                if self.path=='/api/control':
                    if type(payload.get('paused')) is not bool: raise ValueError('paused must be boolean')
                    state.control(payload['paused'])
                elif self.path=='/api/rotate':
                    rot=payload.get('rotate')
                    if type(rot) is not int or rot not in (0,90,180,270):
                        raise ValueError('rotate must be 0, 90, 180, or 270')
                    state.set_rotate(rot)
                elif self.path=='/api/mode':
                    m=payload.get('mode')
                    if m not in ('live','simulation','replay'):
                        raise ValueError('mode must be live, simulation, or replay')
                    state.set_mode(m)
                elif self.path=='/api/rife':
                    factor=payload.get('factor')
                    if type(factor) is not int or not 0 <= factor <= 8:
                        raise ValueError('factor must be an integer between 0 and 8')
                    state.set_rife(factor)
                elif self.path=='/api/memory':
                    ttl=payload.get('memory_ttl_s')
                    if type(ttl) is bool or type(ttl) not in (int, float) or not 0 <= float(ttl) <= 60:
                        raise ValueError('memory_ttl_s must be between 0 and 60')
                    state.set_memory_ttl(float(ttl))
                    return self.send(200,{'ok':True,'memory_ttl_s':state.predictor.memory_ttl_s})
                elif self.path=='/api/depth':
                    d_mode=payload.get('mode')
                    if d_mode not in ('auto','hybrid','pinhole','neural','off'):
                        raise ValueError('depth mode must be auto, hybrid, pinhole, neural, or off')
                    state.set_depth_mode(d_mode)
                    return self.send(200,{'ok':True,'depth_mode':state.depth_estimator.mode})
                elif self.path=='/api/classes':
                    cls_spec=payload.get('classes')
                    if not isinstance(cls_spec, str) or not cls_spec.strip():
                        raise ValueError('classes must be a non-empty preset or comma-separated list')
                    state.set_classes(cls_spec)
                    return self.send(200,{'ok':True,'classes_spec':state.classes_spec,'classes':sorted(resolve_classes(state.classes_spec))})
                elif self.path=='/api/model':
                    mdl_spec=payload.get('model')
                    if not isinstance(mdl_spec, str) or not mdl_spec.strip():
                        raise ValueError('model must be a non-empty detector model name or auto')
                    state.set_model(mdl_spec)
                    return self.send(200,{'ok':True,'model_spec':state.model_spec,'resolved_model':resolve_detector_model(state.model_spec)})
                elif self.path=='/api/camera':
                    cfg=state.set_camera_config(
                        resolution=payload.get('resolution'),
                        quality=payload.get('quality'),
                        stream_mode=payload.get('stream_mode'),
                    )
                    return self.send(200,{'ok':True,**cfg})
                elif self.path=='/api/decision':
                    backend=payload.get('backend')
                    if backend not in DecisionForecaster.VALID_BACKENDS:
                        raise ValueError(f'backend must be one of {DecisionForecaster.VALID_BACKENDS}')
                    state.set_decision_backend(backend)
                elif self.path=='/api/calibration':
                    config=None if payload.get('clear') is True else calibration(payload)
                    if config and state.mode=='live':
                        current=state.snapshot()['frame']
                        if current is None or (config['width'],config['height']) != (current['width'],current['height']):
                            raise ValueError('Calibration must match the displayed camera dimensions.')
                    state.output.mkdir(parents=True,exist_ok=True)
                    temp=state.output/'calibration.tmp';temp.write_text(json.dumps(config,indent=2))
                    temp.replace(state.output/'calibration.json')
                    with state.lock: state.config=config
                else: return self.send(404,{'error':'Not found'})
                self.send(200,{'ok':True})
            except (ValueError,TypeError,KeyError,OSError) as exc:
                self.send(400,{'error':str(exc)})
    return Handler


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--camera-url', default=None, help='Camera HTTP MJPEG (/stream), WebSocket (ws://.../ws), or snapshot (/capture) endpoint')
    p.add_argument('--camera-res', default='320x240', help='ESP32 sensor resolution (default: 320x240 QVGA for ~50-90ms low latency)')
    p.add_argument('--camera-quality', type=int, default=14, help='ESP32 JPEG compression quality 4-63 (default: 14)')
    p.add_argument('--stream-mode', choices=list(VALID_STREAM_MODES), default='auto', help='Camera transport mode: auto, mjpeg, ws, or snapshot')
    p.add_argument('--no-stream', action='store_true', help='Disable persistent MJPEG/WebSocket stream and force HTTP snapshot polling')
    p.add_argument('--replay', type=Path, default=None, help='Dataset directory for replay')
    p.add_argument('--demo', action='store_true', help='Start in simulation mode')
    p.add_argument('--mode', choices=['live', 'simulation', 'replay'], default=None, help='Initial feed mode')
    p.add_argument('--port',type=int,default=8080)
    p.add_argument('--command-port', type=int, default=8002, help='Dedicated v2 motor command listener')
    p.add_argument('--command-bind', default='127.0.0.1', help='Set to GX10 LAN IP for ESP32 access')
    p.add_argument('--command-token-file', type=Path, help='Optional bearer token shared with ESP32')
    p.add_argument('--haptic-near', type=float, default=0.5)
    p.add_argument('--haptic-far', type=float, default=3.0)
    p.add_argument('--haptic-activate', type=float, default=0.65)
    p.add_argument('--haptic-deactivate', type=float, default=0.45)
    p.add_argument('--haptic-rise', type=float, default=0.08)
    p.add_argument('--haptic-fall', type=float, default=0.25)
    p.add_argument('--haptic-max-age', type=float, default=2.5)
    p.add_argument('--model',default='auto',help='Detector model (auto, yolov8l-worldv2.pt, yolo26x.pt, yolo26l.pt, yolov8s-worldv2.pt, yolo26n.pt)')
    p.add_argument('--device',default='0')
    p.add_argument('--classes',default='indoor_nav',help='Obstacle class preset (indoor_nav, full_surround, core) or comma-separated list')
    p.add_argument('--memory-ttl',type=float,default=12.0,help='Persistent radar spatial memory retention in seconds (default: 12.0)')
    p.add_argument('--depth-mode',choices=['auto','hybrid','pinhole','neural','off'],default='auto',help='GX10 monocular metric depth estimation mode (default: auto)')
    p.add_argument('--rotate',type=int,choices=[0,90,180,270],default=None,help='Clockwise rotation before detection (default: 270 for live, 0 for simulation/replay)')
    p.add_argument('--rife-factor',type=int,default=0,help='RIFE optical-flow intermediate frames per capture (default: 0 / Off)')
    p.add_argument('--decision-backend',choices=list(DecisionForecaster.VALID_BACKENDS),default='gemma4:26b',help='Decision model backend (default: gemma4:26b on GX10 Ollama, falls back to laya/kinematic)')
    p.add_argument('--ollama-url',default='http://127.0.0.1:11434',help='Local GX10 Ollama server URL (default: http://127.0.0.1:11434)')
    p.add_argument('--jev-key-file',default='private/jev_api_key.txt',help='Path to private file containing TypeSafe Jev API key')
    p.add_argument('--stream',action='store_true',help='Force continuous multipart MJPEG video stream ingestion')
    p.add_argument('--save-artifacts',action='store_true',help='Save diagnostic artifacts (latest.jpg, latest.json) to disk')
    p.add_argument('--interval',type=float,default=None,help='Delay after each frame; live defaults to 0, replay/demo to 0.5 s')
    p.add_argument('--timeout',type=float,default=3.5,help='Camera socket inactivity timeout in seconds')
    p.add_argument('--max-age',type=float,default=3.5)
    p.add_argument('--output',type=Path,default=Path('artifacts/dashboard'))
    p.add_argument('--calibration',type=Path,help='Explicitly reuse a calibration for the same fixed camera pose')
    args=p.parse_args()
    if args.mode:
        mode = args.mode
    elif args.demo:
        mode = 'simulation'
    elif args.replay:
        mode = 'replay'
    elif args.camera_url:
        mode = 'live'
    else:
        mode = 'simulation'
    if args.camera_url is None and not args.demo and not args.replay:
        args.camera_url = 'http://169.233.72.178/stream'
    if args.rotate is None:
        args.rotate = 270 if mode == 'live' else 0
    if args.interval is None: args.interval=0.01 if args.camera_url else .5
    if (not 1 <= args.port <= 65535 or not all(math.isfinite(v) for v in (args.interval,args.timeout,args.max_age,args.memory_ttl))
            or args.interval < 0 or args.timeout <= 0 or args.max_age <= 0 or not 0 <= args.rife_factor <= 8
            or not 0 <= args.memory_ttl <= 60 or not 4 <= args.camera_quality <= 63):
        p.error('Invalid port, timing, memory TTL, camera quality, or RIFE factor settings.')
    if args.camera_url and urllib.parse.urlsplit(args.camera_url).scheme not in ('http','https','ws','wss'):
        p.error('Camera URL must use HTTP, HTTPS, WS, or WSS.')
    try:
        normalize_camera_resolution(args.camera_res)
        payload=json.loads(args.calibration.read_text()) if args.calibration else None
        if args.calibration and not isinstance(payload,dict):
            p.error('Calibration file is empty or cleared; calibrate again in the dashboard.')
        config=calibration(payload) if payload else None
    except (ValueError,OSError) as exc:
        p.error(str(exc))
    try:
        haptic_settings = HapticSettings(args.haptic_near, args.haptic_far,
                                         args.haptic_activate, args.haptic_deactivate,
                                         args.haptic_rise, args.haptic_fall, args.haptic_max_age)
        if not 1 <= args.command_port <= 65535 or args.command_port == args.port:
            raise ValueError('Command port must be valid and distinct from dashboard port')
        command_token = args.command_token_file.read_text().strip() if args.command_token_file else None
        if args.command_token_file and (not command_token or not command_token.isascii()
                                       or any(c.isspace() for c in command_token)):
            raise ValueError('Command token must be a nonempty ASCII token without whitespace')
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    stream_mode = 'snapshot' if args.no_stream else ('mjpeg' if args.stream else args.stream_mode)
    state=State(
        mode,
        args.output,
        args.max_age,
        config,
        rotate=args.rotate,
        rife_factor=args.rife_factor,
        device=args.device,
        decision_backend=args.decision_backend,
        jev_key_file=args.jev_key_file,
        start_decision_worker=True,
        memory_ttl_s=args.memory_ttl,
        depth_mode=args.depth_mode,
        classes_spec=args.classes,
        model_spec=args.model,
        ollama_url=args.ollama_url,
        camera_resolution=args.camera_res,
        camera_quality=args.camera_quality,
        stream_mode=stream_mode,
        haptic_settings=haptic_settings,
    )
    stop = threading.Event()
    server = command_server = command_thread = worker = None
    try:
        server = ThreadingHTTPServer(('127.0.0.1', args.port), handler_for(state))
        command_server = ThreadingHTTPServer((args.command_bind, args.command_port),
                                             haptic_handler(state.haptic_command, command_token))
        command_thread = threading.Thread(target=command_server.serve_forever, daemon=True)
        command_thread.start()
        print(f'Motor commands: http://{args.command_bind}:{args.command_port}/command', flush=True)
        worker = threading.Thread(target=run_source, args=(state, args, stop), daemon=True)
        worker.start()
        print(f'YOLO dashboard: http://127.0.0.1:{args.port} ({mode})', flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        state.decision_forecaster.stop()
        if command_thread is not None:
            command_server.shutdown()
            command_thread.join(timeout=1)
        if command_server is not None:
            command_server.server_close()
        if server is not None:
            server.server_close()
        if worker is not None:
            worker.join(timeout=1)



if __name__=='__main__': main()
