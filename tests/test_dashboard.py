import json
import tempfile
import threading
import time
import unittest
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer
from pathlib import Path

from software.dashboard.geometry import calibration, project
from software.dashboard.server import State, handler_for


def config():
    return calibration({'width':640,'height':480,'points':[[.1,.9,-1,1],[.9,.9,1,1],[.9,.2,1,3],[.1,.2,-1,3]]})


class DashboardTests(unittest.TestCase):
    def test_ground_projection_and_reject_unsupported_positions(self):
        c=config()
        self.assertEqual(project([300,100,340,264],640,480,c),[0.0,2.0])
        self.assertIsNone(project([300,100,340,479],640,480,c))
        self.assertIsNone(project([0,10,20,100],640,480,c))
        self.assertIsNone(project([300,100,340,264],480,640,c))

    def test_invalid_calibration_is_rejected(self):
        for points in ([[.1,.1,0,1]]*4, [[0,0,-1,1],[1,1,1,1],[1,0,1,3],[0,1,-1,3]],
                       [[0,0,-1,1],[1,0,1,1],[1,1,float('nan'),3],[0,1,-1,3]]):
            with self.assertRaises(ValueError):calibration({'width':640,'height':480,'points':points})

    def test_frame_expiry_failure_pause_and_generation(self):
        state=State('live',Path('/tmp'),max_age=1,config=config())
        frame={'width':640,'height':480,'zones':[0,0,1,1,0,0],
               'objects':[{'class':'person','confidence':.9,'xyxy':[300,100,340,264]}]}
        state.publish(b'image',frame,time.monotonic(),0)
        self.assertEqual(state.snapshot()['objects'][0]['ground_xy'],[0.,2.])
        state.error='capture timed out'
        self.assertFalse(state.snapshot()['valid']);self.assertEqual(state.snapshot()['objects'],[])
        state.publish(b'image',frame,time.monotonic()-2,0)
        self.assertEqual(state.snapshot()['status'],'stale')
        state.control(True);state.publish(b'inflight',frame,time.monotonic(),0)
        self.assertEqual(state.snapshot()['status'],'paused')
        state.control(False)
        self.assertFalse(state.snapshot()['valid'])
        state.publish(b'old',frame,time.monotonic(),0)
        self.assertFalse(state.snapshot()['valid'])
        state.publish(b'new',frame,time.monotonic(),2)
        self.assertTrue(state.snapshot()['valid'])

    def test_http_controls_calibration_and_origin_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            state=State('simulation',directory)
            server=ThreadingHTTPServer(('127.0.0.1',0),handler_for(state))
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            base=f'http://127.0.0.1:{server.server_port}'
            try:
                with urllib.request.urlopen(base+'/') as r:self.assertIn(b'Calibrate floor',r.read())
                def post(path,body,headers=None):
                    request=urllib.request.Request(base+path,json.dumps(body).encode(),headers or {'Content-Type':'application/json'})
                    return urllib.request.urlopen(request)
                with post('/api/control',{'paused':True}):pass
                self.assertTrue(state.paused)
                with post('/api/calibration',config()):pass
                self.assertTrue((Path(directory)/'calibration.json').exists())
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    post('/api/control',{'paused':False},{'Origin':'http://evil.example'})
                self.assertEqual(caught.exception.code,403)
                with self.assertRaises(urllib.error.HTTPError):
                    urllib.request.urlopen(urllib.request.Request(base+'/api/state',headers={'Host':'evil.example'}))
                with post('/api/calibration',{'clear':True}):pass
                self.assertIsNone(state.config)
            finally:server.shutdown();server.server_close();thread.join()


if __name__=='__main__':unittest.main()
