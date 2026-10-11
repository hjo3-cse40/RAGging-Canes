import io
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image
from software.vision import Pipeline, decode_jpeg, make_handler, parse_boundaries, zone_states


class VisionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.pipeline = Pipeline(self.directory.name)
        data = io.BytesIO()
        Image.new("RGB", (640, 480)).save(data, format="JPEG")
        self.jpeg = data.getvalue()

    def tearDown(self):
        self.directory.cleanup()

    def test_center_pair_and_calibrated_boundaries(self):
        self.assertEqual(zone_states([(310, 330)], 640, [i / 6 for i in range(7)]),
                         [0, 0, 1, 1, 0, 0])
        self.assertEqual(zone_states([(0, 30)], 640, [0, .1, .2, .5, .7, .9, 1]),
                         [1, 0, 0, 0, 0, 0])
        with self.assertRaises(ValueError):
            decode_jpeg(b"not an image")
        with self.assertRaises(Exception):
            parse_boundaries("0,.2,.1,.4,.6,.8,1")

    def test_pipeline_rotation_transposes_image_and_artifacts(self):
        rot_pipeline = Pipeline(self.directory.name, rotate=90)
        response = rot_pipeline.process(self.jpeg, "rot-1")
        self.assertEqual(response["width"], 480)
        self.assertEqual(response["height"], 640)
        with Image.open(Path(self.directory.name) / "latest.jpg") as saved:
            self.assertEqual(saved.size, (480, 640))

    def test_receive_only_is_unknown_and_old_results_expire(self):
        response = self.pipeline.process(self.jpeg, "42")
        self.assertEqual(response["zones"], [-1] * 6)
        self.assertEqual(response["status"], "received_only")
        self.pipeline.latest = {"frame_id": "42", "status": "ok", "zones": [1] * 6}
        self.pipeline.latest_started = time.monotonic() - 2
        response = self.pipeline.current_result()
        self.assertEqual(response["status"], "expired")
        self.assertEqual(response["zones"], [-1] * 6)
        self.assertEqual(response["remaining_valid_ms"], 0)

    def test_http_upload_decode_and_busy_rejection(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.pipeline))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}"
        try:
            request = urllib.request.Request(url + "/infer", self.jpeg,
                                            headers={"X-Frame-ID": "7", "Content-Type": "image/jpeg"})
            with urllib.request.urlopen(request) as response:
                result = json.load(response)
            self.assertEqual(result["frame_id"], "7")
            self.assertEqual(result["width"], 640)
            with urllib.request.urlopen(url + "/command") as response:
                command = json.load(response)
            self.assertFalse(command["valid"])
            self.assertEqual(command["zones"], [-1] * 6)
            self.assertEqual(len(command["session_id"]), 32)
            with self.pipeline.state_lock:
                self.pipeline.latest = {"status": "ok", "zones": [0, 0, 1, 1, 0, 0],
                                        "command_sequence": 2}
                self.pipeline.latest_started = time.monotonic()
            with urllib.request.urlopen(url + "/command") as response:
                command = json.load(response)
            self.assertTrue(command["valid"])
            self.assertEqual(command["frame_id"], 2)
            self.assertEqual(command["zones"], [0, 0, 1, 1, 0, 0])
            self.assertGreater(command["ttl_ms"], 0)
            with urllib.request.urlopen(url + "/latest.jpg") as response:
                self.assertEqual(response.read(), self.jpeg)
            bad = urllib.request.Request(url + "/infer", b"bad", headers={"X-Frame-ID": "8"})
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(bad)
            self.assertEqual(error.exception.code, 400)
            self.pipeline.lock.acquire()
            try:
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(request)
                self.assertEqual(error.exception.code, 429)
            finally:
                self.pipeline.lock.release()
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_authenticated_command_listener_does_not_expose_images(self):
        token = "test-token-" + "a" * 32
        server = ThreadingHTTPServer(("127.0.0.1", 0),
                                     make_handler(self.pipeline, token, commands_only=True))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}"
        try:
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(url + "/command")
            self.assertEqual(error.exception.code, 401)
            self.assertIsNone(self.pipeline.last_command_poll)
            headers = {"Authorization": "Bearer " + token}
            with urllib.request.urlopen(urllib.request.Request(url + "/command", headers=headers)) as response:
                self.assertEqual(json.load(response)["zones"], [-1] * 6)
            self.assertIsNotNone(self.pipeline.last_command_poll)
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(urllib.request.Request(url + "/latest.jpg", headers=headers))
            self.assertEqual(error.exception.code, 404)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_idle_pull_waits_for_authorized_command(self):
        captures = []
        jpeg = self.jpeg

        class Camera(BaseHTTPRequestHandler):
            def do_GET(self):
                captures.append(self.path)
                self.send_response(200)
                self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)

            def log_message(self, *args):
                pass

        camera = ThreadingHTTPServer(("127.0.0.1", 0), Camera)
        thread = threading.Thread(target=camera.serve_forever, daemon=True)
        thread.start()
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        token = "test-only-" + "b" * 32
        token_path = Path(self.directory.name) / "token"
        token_path.write_text(token)
        process = subprocess.Popen([
            sys.executable, str(Path(__file__).resolve().parents[1] / "software/vision.py"),
            "pull", "--url", f"http://127.0.0.1:{camera.server_port}/capture",
            "--port", str(port), "--token-file", str(token_path),
            "--idle-without-client", "--frames", "1", "--interval", "0.01",
            "--output", self.directory.name,
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        url = f"http://127.0.0.1:{port}"
        headers = {"Authorization": "Bearer " + token}
        try:
            deadline = time.monotonic() + 5
            while True:
                try:
                    with urllib.request.urlopen(urllib.request.Request(url + "/health", headers=headers), timeout=.3):
                        break
                except OSError:
                    if time.monotonic() >= deadline:
                        self.fail("Pull server did not start")
                    time.sleep(.05)
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(url + "/command")
            self.assertEqual(error.exception.code, 401)
            time.sleep(.5)
            self.assertEqual(captures, [])
            with urllib.request.urlopen(urllib.request.Request(url + "/command", headers=headers)):
                pass
            self.assertEqual(process.wait(timeout=5), 0)
            self.assertEqual(len(captures), 1)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
            camera.shutdown()
            camera.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
