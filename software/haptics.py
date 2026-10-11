"""Three calibrated-strength requests from fresh confidence and distance observations.

This module has no model dependencies. Call Controller under the owning state's lock.
Strength is a requested percentage, not measured vibration amplitude.
"""
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
import hmac
import json
import math
import time
import urllib.parse
import uuid


@dataclass(frozen=True)
class Settings:
    near_m: float = 0.5
    far_m: float = 3.0
    activate: float = 0.65
    deactivate: float = 0.45
    rise_s: float = 0.08
    fall_s: float = 0.25
    max_age_s: float = 2.5

    def __post_init__(self):
        if not all(type(v) in (int, float) and math.isfinite(v) for v in vars(self).values()):
            raise ValueError('Haptic settings must be finite numbers')
        if not (0 < self.near_m < self.far_m and 0 <= self.deactivate < self.activate <= 1
                and self.rise_s > 0 and self.fall_s > 0 and 0 < self.max_age_s <= 2.5):
            raise ValueError('Invalid haptic distance, confidence, smoothing, or age settings')


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


class Controller:
    def __init__(self, settings=None, clock=time.monotonic):
        self.settings = settings or Settings()
        self.clock = clock
        self.session_id = uuid.uuid4().hex
        self.command_id = 0
        self.key = None
        self.last_time = None
        self.backend = None
        self.active = [False] * 3
        self.levels = [0.0] * 3
        self.cached = None
        self.deadline = 0
        self.invalid_reason = None

    def stop(self, reason):
        if self.invalid_reason != reason or self.cached is not None:
            self.command_id += 1
        self.invalid_reason = reason
        self.cached = None
        self.active = [False] * 3
        self.levels = [0.0] * 3
        self.last_time = None
        return self._response(False, reason, [False]*3, [0]*3, [None]*3, [None]*3, 0)

    def _response(self, valid, status, motor_valid, strengths, distances, scores, ttl):
        return dict(schema_version=2, session_id=self.session_id, command_id=self.command_id,
                    valid=valid, status=status, motor_valid_3=list(motor_valid),
                    motor_strength_3=list(strengths), distance_m_3=list(distances),
                    confidence_3=list(scores), ttl_ms=ttl)

    def command(self, decision, generation, enabled=True, reason='unavailable'):
        now = self.clock()
        if not enabled:
            return self.stop(reason)
        obs = (decision or {}).get('haptic_observation')
        if not isinstance(obs, dict) or obs.get('generation') != generation:
            return self.stop('waiting_for_decision')
        started = obs.get('started')
        sequence = obs.get('sequence')
        if not number(started) or not isinstance(sequence, int) or not 0 <= now-started < self.settings.max_age_s:
            return self.stop('stale_decision')
        scores = decision.get('motor_probabilities_3')
        if not isinstance(scores, (list, tuple)) or len(scores) != 3 or not all(number(p) and 0 <= p <= 1 for p in scores):
            return self.stop('invalid_confidence')
        key = (generation, sequence)
        # Re-reading a sample never changes its strength or extends its lifetime.
        if self.key == key:
            if self.cached is None:
                return self.stop('waiting_for_new_observation')
            ttl = max(0, math.floor((self.deadline-now)*1000))
            return {**self.cached, 'ttl_ms': ttl} if ttl else self.stop('expired')
        if self.key is not None and (key < self.key or (self.last_time is not None and started <= self.last_time)):
            return self.stop('out_of_order')
        if self.backend != decision.get('backend'):
            self.active = [False]*3
            self.levels = [0.0]*3
            self.last_time = None
        self.backend = decision.get('backend')
        if self.last_time is not None and started-self.last_time >= self.settings.max_age_s:
            self.active = [False]*3
            self.levels = [0.0]*3
            self.last_time = None
        dt = max(0, started-self.last_time) if self.last_time is not None else None
        self.key, self.last_time = key, started
        self.deadline = started + self.settings.max_age_s
        self.command_id += 1
        self.invalid_reason = None
        distances, motor_valid = [None]*3, [True]*3
        objects = obs.get('objects')
        if not isinstance(objects, list):
            return self.stop('invalid_observation')
        eligible = [o for o in objects if o.get('realistic') is True and not o.get('remembered')]
        if decision.get('realistic_detection') is not True:
            eligible = []
        for i, zone in enumerate(('LEFT', 'CENTER', 'RIGHT')):
            targets = [o for o in eligible if o.get('current_zone_3') == zone]
            depths = [o.get('depth_m') for o in targets]
            # If any target has unknown depth, we cannot identify the nearest one.
            if any(not number(d) or d <= 0 for d in depths):
                motor_valid[i] = False
            elif depths:
                distances[i] = min(depths)
            if not motor_valid[i] or not targets:
                self.active[i], self.levels[i] = False, 0.0
                continue
            p = scores[i]
            if p >= self.settings.activate:
                self.active[i] = True
            elif p <= self.settings.deactivate:
                self.active[i] = False
            proximity = max(0, min(1, (self.settings.far_m-distances[i]) / (self.settings.far_m-self.settings.near_m)))
            target = 100*p*proximity if self.active[i] else 0
            tau = self.settings.rise_s if target > self.levels[i] else self.settings.fall_s
            alpha = 1 if dt is None else 1-math.exp(-dt/tau)
            self.levels[i] += alpha*(target-self.levels[i])
        self.cached = self._response(True, 'ok' if all(motor_valid) else 'partial', motor_valid,
                                    [round(v) for v in self.levels], distances, scores, 0)
        self.cached['source_sequence'] = sequence
        self.cached['source_mode'] = obs.get('source_mode', 'unknown')
        ttl = max(0, math.floor((self.deadline-now)*1000))
        return {**self.cached, 'ttl_ms': ttl} if ttl else self.stop('expired')


def handler_for(command, token=None):
    """Dedicated read-only listener: no dashboard controls or image endpoints."""
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if token and not hmac.compare_digest(self.headers.get('Authorization', '').encode(), ('Bearer '+token).encode()):
                self.send_error(401)
                return
            if urllib.parse.urlsplit(self.path).path != '/command':
                self.send_error(404)
                return
            body = json.dumps(command(), allow_nan=False).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass
    return Handler
