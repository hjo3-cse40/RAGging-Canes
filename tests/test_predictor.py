import io
import unittest
from pathlib import Path
import tempfile

from PIL import Image, ImageDraw
from software.predictor import RIFEInterpolator, TrajectoryPredictor
from software.dashboard.geometry import calibration
from software.dashboard.server import State


def make_test_jpeg(offset_x=100, bg_color=(30, 40, 50)):
    im = Image.new("RGB", (320, 240), bg_color)
    draw = ImageDraw.Draw(im)
    draw.rectangle((offset_x, 60, offset_x + 50, 200), fill=(0, 240, 200))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


class PredictorAndRIFETests(unittest.TestCase):
    def test_rife_interpolator_synthesizes_intermediate_frames_and_gates_scene_cuts(self):
        rife = RIFEInterpolator(factor=3, device="cpu", scene_cut_threshold=0.20)
        f1 = make_test_jpeg(offset_x=80)
        f2 = make_test_jpeg(offset_x=96)

        # First frame has no predecessor -> returns 1 frame (the raw frame)
        out1, meta1 = rife.push_and_interpolate_jpegs(f1)
        self.assertEqual(len(out1), 1)
        self.assertEqual(meta1["interpolated_count"], 0)

        # Second frame with realistic motion -> 3 intermediate + 1 real = 4 frames
        out2, meta2 = rife.push_and_interpolate_jpegs(f2)
        self.assertEqual(len(out2), 4)
        self.assertEqual(meta2["interpolated_count"], 3)
        self.assertFalse(meta2["scene_cut_skipped"])
        self.assertTrue(all(isinstance(item[0], bytes) and len(item[0]) > 100 for item in out2))

        # Abrupt head-whip / scene cut -> gated out to prevent ghosting
        f_cut = make_test_jpeg(offset_x=220, bg_color=(250, 250, 250))
        out3, meta3 = rife.push_and_interpolate_jpegs(f_cut)
        self.assertEqual(len(out3), 1)
        self.assertEqual(meta3["interpolated_count"], 0)
        self.assertTrue(meta3["scene_cut_skipped"])

    def test_trajectory_predictor_rolls_out_3s_horizon_and_computes_ttc(self):
        calib = calibration({
            "width": 640,
            "height": 480,
            "points": [[0.1, 0.9, -1, 1], [0.9, 0.9, 1, 1], [0.9, 0.2, 1, 3], [0.1, 0.2, -1, 3]],
        })
        pred = TrajectoryPredictor()

        # Frame 1: Person on left-ish moving right and closer (y2 increasing)
        objs1 = [{"class": "person", "confidence": 0.92, "xyxy": [140, 140, 200, 300]}]
        res1, scen1 = pred.update(objs1, 640, 480, timestamp=10.0, calibration_config=calib)
        self.assertTrue(scen1["realistic"])
        self.assertEqual(len(res1), 1)

        # Frame 2 (t=10.2s): Person moved right (+24px -> 120 px/s) and closer (+16px y2)
        objs2 = [{"class": "person", "confidence": 0.93, "xyxy": [164, 144, 228, 316]}]
        res2, scen2 = pred.update(objs2, 640, 480, timestamp=10.2, calibration_config=calib)
        self.assertTrue(scen2["realistic"])
        self.assertEqual(res2[0]["hits"], 2)
        self.assertIn("2.0s", res2[0]["predicted_boxes"])
        self.assertEqual(len(res2[0]["predicted_path"]), 4)
        # Future X at 2.0s should be further right than current X
        self.assertGreater(res2[0]["predicted_boxes"]["2.0s"][0], 164)
        self.assertIsNotNone(res2[0]["ttc_s"])
        self.assertGreaterEqual(sum(scen2["predicted_zones_2s"]), 1)

    def test_realism_filter_rejects_impossible_teleport_and_transient_flicker(self):
        pred = TrajectoryPredictor()

        # 1. Low-confidence 1-frame blip is marked transient_unconfirmed
        blip = [{"class": "person", "confidence": 0.38, "xyxy": [50, 100, 90, 220]}]
        res_blip, scen_blip = pred.update(blip, 640, 480, timestamp=1.0)
        self.assertFalse(res_blip[0]["realistic"])
        self.assertEqual(res_blip[0]["realism_reason"], "transient_unconfirmed")
        self.assertFalse(scen_blip["realistic"])

        # 2. Impossible teleport jump (500px in 0.05s = 10,000 px/s) is flagged unrealistic
        pred.reset()
        pred.update([{"class": "person", "confidence": 0.9, "xyxy": [40, 100, 90, 260]}], 640, 480, timestamp=2.0)
        res_jump, scen_jump = pred.update(
            [{"class": "person", "confidence": 0.9, "xyxy": [240, 100, 290, 260]}],
            640, 480, timestamp=2.02
        )
        self.assertFalse(res_jump[0]["realistic"])
        self.assertEqual(res_jump[0]["realism_reason"], "unrealistic_velocity_jump")
        self.assertFalse(scen_jump["realistic"])

    def test_dashboard_state_exposes_scenario_3zones_and_controls(self):
        import time
        with tempfile.TemporaryDirectory() as tmp:
            state = State("live", Path(tmp), rife_factor=2, device="cpu")
            f1 = make_test_jpeg(100)
            f2 = make_test_jpeg(112)
            now = time.monotonic()
            state.publish(f1, {"width": 320, "height": 240, "zones": [0, 1, 0, 0, 0, 0],
                               "objects": [{"class": "person", "confidence": 0.9, "xyxy": [100, 60, 150, 200]}]},
                          now - 0.2, 0)
            state.publish(f2, {"width": 320, "height": 240, "zones": [0, 0, 1, 0, 0, 0],
                               "objects": [{"class": "person", "confidence": 0.92, "xyxy": [112, 60, 162, 200]}]},
                          now, 0)
            snap = state.snapshot()
            self.assertIn("scenario", snap)
            self.assertTrue(snap["scenario"]["realistic"])
            self.assertEqual(snap["zones_3"], [0, 1, 0])
            self.assertEqual(len(snap["predicted_zones_3_2s"]), 3)
            self.assertEqual(snap["rife"]["interpolated_count"], 2)
            state.set_rife(0)
            self.assertEqual(state.interpolator.factor, 0)
            state.set_decision_backend("kinematic")
            self.assertEqual(state.decision_forecaster.requested_backend, "kinematic")
            state.decision_forecaster.stop()
            with self.assertRaises(ValueError):
                state.set_rife(99)
            with self.assertRaises(ValueError):
                state.set_decision_backend("invalid_model")

    def test_decision_forecaster_laya_and_jev_api_key_file(self):
        import json
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from software.predictor import DecisionForecaster, six_to_three_zones, three_to_six_zones

        self.assertEqual(six_to_three_zones([0, 0, 1, 1, 0, 0]), [0, 1, 0])
        self.assertEqual(six_to_three_zones([1, 0, 0, 0, 0, 1]), [1, 0, 1])
        self.assertEqual(three_to_six_zones([0, 1, 0]), [0, 0, 1, 1, 0, 0])

        pred = TrajectoryPredictor()
        pred.update([{"class": "person", "confidence": 0.93, "xyxy": [200, 120, 280, 340]}], 640, 480, timestamp=1.0)
        _, scen = pred.update([{"class": "person", "confidence": 0.95, "xyxy": [260, 110, 360, 390]}], 640, 480, timestamp=1.5)

        # 1. Test local Laya (or kinematic fallback if laya is not installed in current runner)
        forecaster_laya = DecisionForecaster(backend="laya")
        dec_laya = forecaster_laya.evaluate_sync(scen)
        self.assertIn(dec_laya["backend"], ("laya", "kinematic"))
        self.assertTrue(dec_laya["realistic_detection"])
        self.assertEqual(len(dec_laya["recommended_pre_alert_zones_3"]), 3)
        self.assertEqual(len(dec_laya["recommended_pre_alert_zones"]), 6)

        # 2. Test Jev API reading private key file and calling /v1/systemone
        received_headers = {}
        received_body = {}

        class MockJevHandler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                received_headers["auth"] = self.headers.get("Authorization")
                length = int(self.headers.get("Content-Length", "0"))
                received_body.update(json.loads(self.rfile.read(length)))
                resp = {
                    "model": "jev-latest",
                    "answers": {
                        "hazard_decision": {
                            "type": "choice",
                            "choice": "approaching_center",
                            "confidence": 0.96,
                            "probabilities": {
                                "approaching_center": 0.96,
                                "crossing_left_to_center": 0.02,
                                "crossing_right_to_center": 0.01,
                                "left_obstacle": 0.005,
                                "right_obstacle": 0.003,
                                "unrealistic_or_clear": 0.002,
                            },
                        }
                    },
                    "usage": {"input_tokens": 90, "output_tokens": 0},
                }
                raw = json.dumps(resp).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        server = ThreadingHTTPServer(("127.0.0.1", 0), MockJevHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                key_path = Path(tmp) / "jev_api_key.txt"
                key_path.write_text("jev-test-secret-key-123\n", encoding="utf-8")
                forecaster_jev = DecisionForecaster(
                    backend="jev",
                    jev_key_file=key_path,
                    jev_url=f"http://127.0.0.1:{server.server_port}/v1/systemone",
                )
                dec_jev = forecaster_jev.evaluate_sync(scen)
                self.assertEqual(received_headers.get("auth"), "Bearer jev-test-secret-key-123")
                self.assertIn("hazard_decision", received_body.get("questions", {}))
                self.assertEqual(dec_jev["backend"], "jev")
                self.assertTrue(dec_jev["realistic_detection"])
                self.assertEqual(dec_jev["motion_trend"], "approaching_center")
                self.assertEqual(dec_jev["recommended_pre_alert_zones_3"], [0, 1, 0])
                self.assertEqual(dec_jev["recommended_pre_alert_zones"], [0, 0, 1, 1, 0, 0])
        finally:
            server.shutdown()
            server.server_close()
            t.join(timeout=1)


if __name__ == "__main__":
    unittest.main()

