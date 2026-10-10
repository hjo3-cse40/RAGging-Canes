import json
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

from gx10.service import State, handler_for, validate_zones


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.state = State(max_age_ms=2500, clock=lambda: self.now)

    def test_expiry_counts_capture_and_inference_time(self):
        self.state.publish(98.7, [0, 1, -1, 0, 1], {})
        result = self.state.snapshot()
        self.assertTrue(result["valid"])
        self.assertLessEqual(result["ttl_ms"], 1200)
        self.now = 101.3
        result = self.state.snapshot()
        self.assertEqual(result["status"], "stale")
        self.assertEqual(result["zones"], [-1] * 5)
        self.assertEqual(result["ttl_ms"], 0)

    def test_repeated_polls_do_not_refresh_frame(self):
        self.state.publish(100, [1, 0, 0, 0, 0], {})
        first = self.state.snapshot("camera")
        self.now += 1
        second = self.state.snapshot("camera")
        self.assertEqual(first["frame_id"], second["frame_id"])
        self.assertLess(second["ttl_ms"], first["ttl_ms"])
        self.assertEqual(second["poll_count"], 2)

    def test_failure_clears_old_detection_then_recovers(self):
        self.state.publish(100, [1] * 5, {})
        self.state.fail(TimeoutError())
        failed = self.state.snapshot()
        self.assertFalse(failed["valid"])
        self.assertEqual(failed["zones"], [-1] * 5)
        self.state.publish(100, [0] * 5, {})
        self.assertTrue(self.state.snapshot()["valid"])
        self.assertEqual(self.state.snapshot()["frame_id"], 2)

    def test_rejects_truncation_wrong_shape_booleans_and_floats(self):
        for text in ("[1,0]", "[true,0,0,0,0]", "[1.0,0,0,0,0]", "[2,0,0,0,0]", "{\"zones\":[0,0,0,0,0]}"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                validate_zones({"done": True, "done_reason": "stop", "message": {"content": text}})
        with self.assertRaises(ValueError):
            validate_zones({"done": True, "done_reason": "length", "message": {"content": "[0,0,0,0,0]"}})
        self.assertEqual(validate_zones({"done": True, "done_reason": "stop", "message": {"content": "[-1,0,1,0,0]"}}), [-1, 0, 1, 0, 0])

    def test_http_contract_starting_success_and_failure(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(self.state))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/command"
            with urllib.request.urlopen(url) as r:
                self.assertEqual(r.headers["Cache-Control"], "no-store")
                self.assertEqual(json.load(r)["status"], "starting")
            self.state.publish(100, [1, 0, 0, 0, 0], {})
            with urllib.request.urlopen(url) as r:
                self.assertEqual(json.load(r)["zones"], [1, 0, 0, 0, 0])
            self.state.fail(ValueError())
            with urllib.request.urlopen(url) as r:
                self.assertFalse(json.load(r)["valid"])
            with urllib.request.urlopen(url.replace("/command", "/firmware/ZoneClient.h")) as r:
                self.assertIn(b"class ZoneClient", r.read())
                self.assertIn("attachment", r.headers["Content-Disposition"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
