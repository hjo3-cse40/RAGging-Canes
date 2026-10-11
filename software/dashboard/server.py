"""Loopback-only dashboard. Use an SSH tunnel to inspect a remote GX10."""
import argparse
from collections import deque
import base64
import io
import http.client
import json
import math
from pathlib import Path
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image, ImageDraw
from software.vision import Pipeline, MAX_BYTES, decode_jpeg, zone_states
from .geometry import calibration, project

STATIC = Path(__file__).parent


class StreamBroadcaster:
    """Thread-safe fan-out broadcaster for live MJPEG stream clients."""
    def __init__(self):
        self._lock = threading.Lock()
        self._subscribers = []
        self._latest_jpeg = None
        self._cond = threading.Condition(self._lock)

    def subscribe(self):
        q = deque(maxlen=2)
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


class State:
    def __init__(self, mode, output, max_age=2.5, config=None, rotate=0):
        self.lock = threading.Lock()
        self.mode, self.output, self.max_age = mode, Path(output), max_age
        self.rotate = rotate
        self.frame = None
        self.started = None
        self.error = None
        self.paused = False
        self.config = config
        self.generation = 0
        self.arrivals = deque(maxlen=30)
        self.broadcaster = StreamBroadcaster()

    def publish(self, data, result, started, generation):
        with self.lock:
            if self.paused or generation != self.generation:
                return
            self.frame = {**result, 'image': base64.b64encode(data).decode('ascii')}
            self.started, self.error = started, None
            self.arrivals.append(time.monotonic())
        self.broadcaster.broadcast(data)

    def snapshot(self):
        with self.lock:
            frame = dict(self.frame) if self.frame else None
            age = time.monotonic()-self.started if self.started is not None else None
            valid = bool(frame and age is not None and not self.error and not self.paused and age < self.max_age and frame.get('status','ok') == 'ok')
            status = 'paused' if self.paused else 'error' if self.error else 'waiting' if frame is None or age is None else 'live' if valid else 'stale'
            config = self.config
            objects = []
            if valid:
                for obj in frame.get('objects', []):
                    xy = project(obj['xyxy'], frame['width'], frame['height'], config) if config else None
                    objects.append({**obj, 'ground_xy': xy, 'source': 'yolo',
                                    'position_kind': 'floor_estimate' if xy else 'bearing_only'})
            return {'mode': self.mode, 'status': status, 'valid': valid,
                    'error': self.error, 'age_ms': round(age*1000) if age is not None else None,
                    'max_age_ms': round(self.max_age*1000), 'frame': frame,
                    'capture_fps': round((len(self.arrivals)-1)/(self.arrivals[-1]-self.arrivals[0]),2)
                    if valid and len(self.arrivals)>1 and self.arrivals[-1]>self.arrivals[0] else 0,
                    'objects': objects, 'zones': frame['zones'] if valid else [-1]*6,
                    'calibration': config, 'map_mode': 'floor_estimate' if config else 'bearing_only',
                    'rotate': self.rotate}

    def control(self, paused):
        with self.lock:
            self.paused = paused
            self.generation += 1
            self.arrivals.clear()
            # A resume must wait for a newly captured frame.
            self.started = None

    def set_rotate(self, rotate):
        with self.lock:
            if rotate not in (0, 90, 180, 270):
                raise ValueError('Rotation must be 0, 90, 180, or 270')
            self.rotate = rotate
            self.generation += 1
            self.arrivals.clear()
            self.started = None

    def set_mode(self, mode):
        with self.lock:
            if mode not in ('live', 'simulation', 'replay'):
                raise ValueError('Mode must be live, simulation, or replay')
            self.mode = mode
            self.rotate = 270 if mode == 'live' else 0
            self.generation += 1
            self.arrivals.clear()
            self.started = None
            self.error = None



def demo_frame(index):
    im = Image.new('RGB', (640,480), '#d8e0e4'); draw = ImageDraw.Draw(im)
    draw.rectangle((0,240,640,480), fill='#b7c2c5')
    for line_x in range(-640,1280,120): draw.line((320,240,line_x,480),fill='#a0adb3',width=2)
    # Smooth sinusoidal human running pace across sensor field
    # 60 frames per roundtrip (~2.4s at 25 fps, human running speed ~2.5 m/s)
    phase = ((index % 60) / 60.0) * 2 * math.pi
    x = int(280 + 200 * math.sin(phase))
    draw.ellipse((x+20,125,x+60,165),fill='#3f5d70')
    draw.rounded_rectangle((x,168,x+80,340),radius=15,fill='#3f5d70')
    draw.rectangle((x+5,330,x+28,410),fill='#3f5d70');draw.rectangle((x+52,330,x+75,410),fill='#3f5d70')
    draw.text((20,20),'SIMULATION / NOT A CAMERA CAPTURE',fill='#273947')
    data=io.BytesIO();im.save(data,format='JPEG')
    objects=[{'class':'person','confidence':.94,'xyxy':[x,125,x+80,410]}]
    return data.getvalue(), {'width':640,'height':480,'objects':objects,
                             'zones':zone_states([(x,x+80)],640,[i/6 for i in range(7)])}


class SnapshotCamera:
    """Reuse the camera HTTP connection; reconnect only when it closes or fails."""
    def __init__(self, url, timeout):
        self.url = urllib.parse.urlsplit(url)
        self.timeout = timeout
        self.cls = http.client.HTTPSConnection if self.url.scheme == 'https' else http.client.HTTPConnection
        self.connection = self.cls(self.url.hostname, self.url.port, timeout=timeout)

    def _request_once(self):
        query = self.url.query + ('&' if self.url.query else '') + '_cb=' + str(time.time_ns())
        path = urllib.parse.urlunsplit(('', '', self.url.path or '/', query, ''))
        self.connection.request('GET', path, headers={'Cache-Control': 'no-cache', 'Accept': 'image/jpeg'})
        response = self.connection.getresponse()
        if response.status != 200:
            raise OSError(f'Camera HTTP {response.status}')
        data = response.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise ValueError('Camera JPEG exceeds the size limit')
        response.close()
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
            raise

    def close(self):
        self.connection.close()


class StreamCamera:
    """Reads continuous multipart MJPEG stream over a single persistent connection."""
    def __init__(self, url, timeout=3.5):
        self.url_str = url
        self.url = urllib.parse.urlsplit(url)
        self.timeout = timeout
        self.response = None
        self.buffer = bytearray()
        self._connect()

    def _connect(self):
        self.close()
        req = urllib.request.Request(
            self.url_str,
            headers={'Cache-Control': 'no-cache', 'Accept': 'multipart/x-mixed-replace, image/jpeg, */*'}
        )
        self.response = urllib.request.urlopen(req, timeout=self.timeout)
        self.buffer.clear()

    def read(self):
        while True:
            soi = self.buffer.find(b'\xff\xd8')
            if soi != -1:
                eoi = self.buffer.find(b'\xff\xd9', soi + 2)
                if eoi != -1:
                    frame = bytes(self.buffer[soi:eoi + 2])
                    del self.buffer[:eoi + 2]
                    if len(frame) > MAX_BYTES:
                        raise ValueError('Camera JPEG exceeds the size limit')
                    return frame
            read_fn = getattr(self.response, 'read1', None)
            if read_fn is None and hasattr(self.response, 'fp') and hasattr(self.response.fp, 'read1'):
                read_fn = self.response.fp.read1
            if read_fn is None:
                read_fn = self.response.read
            chunk = read_fn(8192)
            if not chunk:
                raise OSError('Camera stream closed')
            self.buffer.extend(chunk)

    def close(self):
        if self.response:
            try: self.response.close()
            except Exception: pass
            self.response = None
        self.buffer.clear()


def create_camera(url, timeout, is_stream=None):
    """Instantiate camera reader: StreamCamera for MJPEG streams, SnapshotCamera for pulls."""
    if is_stream is True or (is_stream is None and ('/stream' in url or 'mjpeg' in url.lower() or 'mjpg' in url.lower())):
        try:
            return StreamCamera(url, timeout)
        except Exception:
            return SnapshotCamera(url, timeout)
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
    if not rotate:
        return raw_bytes, None, None, None
    try:
        import cv2
        import numpy as np
        nparr = np.frombuffer(raw_bytes, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr is not None:
            if rotate == 90:
                bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
            elif rotate == 180:
                bgr = cv2.rotate(bgr, cv2.ROTATE_180)
            elif rotate == 270:
                bgr = cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
            h, w = bgr.shape[:2]
            _, enc = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
            return enc.tobytes(), bgr, w, h
    except (ImportError, Exception):
        pass
    # Pillow fallback
    image = decode_jpeg(raw_bytes).rotate(-rotate, expand=True)
    output = io.BytesIO()
    image.save(output, format='JPEG', quality=88)
    return output.getvalue(), image, image.width, image.height


def run_source(state, args, stop):
    pipeline = None
    camera = None
    producer_thread = None
    buffer = LatestFrameBuffer()
    try:
        if args.camera_url:
            camera = create_camera(args.camera_url, args.timeout, getattr(args, 'stream', None))
            pipeline = Pipeline(
                args.output / 'yolo',
                detect=True,
                model=args.model,
                device=args.device,
                max_result_age_ms=args.max_age * 1000,
                save_disk=getattr(args, 'save_artifacts', False)
            )

            # Dedicated producer thread reads camera continuously without blocking YOLO
            def producer_loop():
                consecutive_errors = 0
                while not stop.is_set():
                    with state.lock:
                        paused = state.paused
                    if paused:
                        if camera: camera.close()
                        stop.wait(0.1)
                        continue
                    started = time.monotonic()
                    try:
                        raw = camera.read()
                        fetched = time.monotonic()
                        buffer.put((raw, started, fetched))
                        consecutive_errors = 0
                    except Exception as exc:
                        consecutive_errors += 1
                        with state.lock:
                            if consecutive_errors >= 2:
                                state.error = str(exc)
                        stop.wait(0.35 if consecutive_errors else 0.01)
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
                paused, generation, rotate, current_mode = state.paused, state.generation, state.rotate, state.mode
            if paused:
                buffer.clear()
                stop.wait(0.1)
                continue

            try:
                if current_mode == 'live' and args.camera_url:
                    item, _, gen = buffer.get(last_gen, timeout=0.1)
                    if item is None or gen == last_gen:
                        continue
                    last_gen = gen
                    raw, started, fetched = item
                    rotated_jpeg, image_input, w, h = prepare_frame_and_rotate(raw, rotate)
                    result = pipeline.process(
                        rotated_jpeg,
                        str(index + 1),
                        observation_started=started,
                        image_input=image_input
                    )
                    result['capture_ms'] = round((fetched - started) * 1000, 2)
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

                result['frame_id'] = index + 1
                state.publish(raw_to_publish, result, started, generation)
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
                            jpeg = q.pop() if q else None
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
    p.add_argument('--camera-url', default=None, help='Camera HTTP JPEG/stream endpoint')
    p.add_argument('--replay', type=Path, default=None, help='Dataset directory for replay')
    p.add_argument('--demo', action='store_true', help='Start in simulation mode')
    p.add_argument('--mode', choices=['live', 'simulation', 'replay'], default=None, help='Initial feed mode')
    p.add_argument('--port',type=int,default=8080)
    p.add_argument('--model',default='yolo26n.pt');p.add_argument('--device',default='0')
    p.add_argument('--rotate',type=int,choices=[0,90,180,270],default=None,help='Clockwise rotation before detection (default: 270 for live, 0 for simulation/replay)')
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
        args.camera_url = 'http://169.233.72.178/capture'
    if args.rotate is None:
        args.rotate = 270 if mode == 'live' else 0
    if args.interval is None: args.interval=0.04 if args.camera_url else .5
    if (not 1 <= args.port <= 65535 or not all(math.isfinite(v) for v in (args.interval,args.timeout,args.max_age))
            or args.interval < 0 or args.timeout <= 0 or args.max_age <= 0):
        p.error('Invalid port or timing settings.')
    if args.camera_url and urllib.parse.urlsplit(args.camera_url).scheme not in ('http','https'):
        p.error('Camera URL must use HTTP or HTTPS.')
    try:
        payload=json.loads(args.calibration.read_text()) if args.calibration else None
        if args.calibration and not isinstance(payload,dict):
            p.error('Calibration file is empty or cleared; calibrate again in the dashboard.')
        config=calibration(payload) if payload else None
    except (ValueError,OSError) as exc:
        p.error(str(exc))
    state=State(mode,args.output,args.max_age,config,rotate=args.rotate);stop=threading.Event()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),handler_for(state))
    worker=threading.Thread(target=run_source,args=(state,args,stop),daemon=True);worker.start()
    print(f'YOLO dashboard: http://127.0.0.1:{args.port} ({mode})',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        stop.set();server.server_close();worker.join(timeout=1)


if __name__=='__main__': main()
