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


class State:
    def __init__(self, mode, output, max_age=2.5, config=None):
        self.lock = threading.Lock()
        self.mode, self.output, self.max_age = mode, Path(output), max_age
        self.frame = None
        self.started = None
        self.error = None
        self.paused = False
        self.config = config
        self.generation = 0
        self.arrivals = deque(maxlen=30)

    def publish(self, data, result, started, generation):
        with self.lock:
            if self.paused or generation != self.generation:
                return
            self.frame = {**result, 'image': base64.b64encode(data).decode('ascii')}
            self.started, self.error = started, None
            self.arrivals.append(time.monotonic())

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
                    'calibration': config, 'map_mode': 'floor_estimate' if config else 'bearing_only'}

    def control(self, paused):
        with self.lock:
            self.paused = paused
            self.generation += 1
            self.arrivals.clear()
            # A resume must wait for a newly captured frame.
            self.started = None


def demo_frame(index):
    im = Image.new('RGB', (640,480), '#d8e0e4'); draw = ImageDraw.Draw(im)
    draw.rectangle((0,240,640,480), fill='#b7c2c5')
    for x in range(-640,1280,120): draw.line((320,240,x,480),fill='#a0adb3',width=2)
    x = [90,160,240,320,410,480,410,320,240,160][index % 10]
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
        self.url=urllib.parse.urlsplit(url)
        cls=http.client.HTTPSConnection if self.url.scheme=='https' else http.client.HTTPConnection
        self.connection=cls(self.url.hostname,self.url.port,timeout=timeout)

    def read(self):
        query=self.url.query+('&' if self.url.query else '')+'_cb='+str(time.time_ns())
        path=urllib.parse.urlunsplit(('', '', self.url.path or '/', query, ''))
        try:
            self.connection.request('GET',path,headers={'Cache-Control':'no-cache'})
            response=self.connection.getresponse()
            if response.status != 200:
                raise OSError(f'Camera HTTP {response.status}')
            data=response.read(MAX_BYTES+1)
            if len(data)>MAX_BYTES:
                raise ValueError('Camera JPEG exceeds the size limit')
            response.close()
            return data
        except Exception:
            self.close()
            raise

    def close(self):
        self.connection.close()


def run_source(state, args, stop):
    pipeline = None
    camera = None
    try:
        if args.camera_url:
            camera=SnapshotCamera(args.camera_url,args.timeout)
            pipeline = Pipeline(args.output/'yolo', detect=True, model=args.model, device=args.device, max_result_age_ms=args.max_age*1000)
        replay = []
        if args.replay:
            captures = [json.loads(line) for line in (args.replay/'captures.jsonl').read_text().splitlines()]
            results = [json.loads(line) for line in (args.replay/'results.jsonl').read_text().splitlines()]
            by_id = {r['frame']: r for r in results if r['backend']=='yolo' and r['repeat']==1 and r.get('valid')}
            for c in captures:
                if c.get('valid') and c['frame'] in by_id:
                    path = (args.replay/c['file']).resolve()
                    if not path.is_relative_to(args.replay.resolve()):
                        raise ValueError('Replay image must be inside its dataset directory.')
                    data = path.read_bytes(); image = decode_jpeg(data)
                    replay.append((data, {**by_id[c['frame']], 'width':image.width, 'height':image.height,
                                          'source_frame':c['file']}))
            if not replay: raise ValueError('No paired YOLO frames in this replay directory.')
        index = 0
        while not stop.is_set():
            with state.lock:
                paused, generation = state.paused, state.generation
            if paused:
                if camera: camera.close()
                stop.wait(.1);continue
            started = time.monotonic()
            try:
                if args.camera_url:
                    raw=camera.read()
                    image=decode_jpeg(raw)
                    if args.rotate:
                        image=image.rotate(-args.rotate,expand=True)
                        output=io.BytesIO();image.save(output,format='JPEG',quality=95);raw=output.getvalue()
                    fetched=time.monotonic()
                    result=pipeline.process(raw,str(index+1),observation_started=started)
                    result['capture_ms']=round((fetched-started)*1000,2)
                elif replay:
                    raw,result=replay[index % len(replay)]
                    result=dict(result)
                else:
                    raw,result=demo_frame(index)
                result['frame_id']=index+1
                state.publish(raw,result,started,generation)
                index+=1
            except Exception as exc:
                with state.lock:
                    if generation == state.generation: state.error=str(exc)
            # Retry failed transfers without a tight reconnect loop.
            stop.wait(max(args.interval, .1) if state.error else args.interval)
    except Exception as exc:
        with state.lock: state.error=str(exc)
    finally:
        if camera: camera.close()


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
            if path=='/api/state':
                snapshot=state.snapshot()
                after=urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query).get('after',[None])[0]
                if snapshot['frame'] and after == str(snapshot['frame']['frame_id']):
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
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--camera-url');source.add_argument('--replay',type=Path);source.add_argument('--demo',action='store_true')
    p.add_argument('--port',type=int,default=8080)
    p.add_argument('--model',default='yolo26n.pt');p.add_argument('--device',default='0')
    p.add_argument('--rotate',type=int,choices=[0,90,180,270],default=0,help='Clockwise rotation before detection')
    p.add_argument('--interval',type=float,default=None,help='Delay after each frame; live defaults to 0, replay/demo to 0.5 s')
    p.add_argument('--timeout',type=float,default=1.5,help='Camera socket inactivity timeout in seconds')
    p.add_argument('--max-age',type=float,default=2.5)
    p.add_argument('--output',type=Path,default=Path('artifacts/dashboard'))
    p.add_argument('--calibration',type=Path,help='Explicitly reuse a calibration for the same fixed camera pose')
    args=p.parse_args()
    if args.interval is None: args.interval=0 if args.camera_url else .5
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
    mode='live' if args.camera_url else 'replay' if args.replay else 'simulation'
    state=State(mode,args.output,args.max_age,config);stop=threading.Event()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),handler_for(state))
    worker=threading.Thread(target=run_source,args=(state,args,stop),daemon=True);worker.start()
    print(f'YOLO dashboard: http://127.0.0.1:{args.port} ({mode})',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        stop.set();server.server_close();worker.join(timeout=1)


if __name__=='__main__': main()
