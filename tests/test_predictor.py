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

    def test_dashboard_state_exposes_scenario_and_rife_controls(self):
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
            self.assertEqual(snap["rife"]["interpolated_count"], 2)
            state.set_rife(0)
            self.assertEqual(state.interpolator.factor, 0)
            with self.assertRaises(ValueError):
                state.set_rife(99)


if __name__ == "__main__":
    unittest.main()
