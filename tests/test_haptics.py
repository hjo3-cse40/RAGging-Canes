import json
from pathlib import Path
import threading
import time
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch
from http.server import ThreadingHTTPServer

from software.haptics import Controller, Settings, handler_for
from software.dashboard.server import State
from software.predictor import DecisionForecaster


def decision(started=10.0, sequence=1, scores=None, depths=(1.5, .5, 3.0), generation=0):
    return {
        'requested_backend': 'kinematic', 'backend': 'kinematic',
        'realistic_detection': True, 'motor_probabilities_3': scores or [.9, .9, .9],
        'haptic_observation': {
            'started': started, 'sequence': sequence, 'generation': generation, 'source_mode': 'live',
            'objects': [dict(current_zone_3=z, depth_m=d, realistic=True, remembered=False)
                        for z, d in zip(('LEFT', 'CENTER', 'RIGHT'), depths)],
        },
    }


class HapticTests(unittest.TestCase):
    def setUp(self):
        self.now = 10.0
        self.controller = Controller(clock=lambda: self.now)

    def test_strength_increases_with_closeness_and_confidence(self):
        result = self.controller.command(decision(), 0)
        self.assertEqual(result['motor_strength_3'], [54, 90, 0])
        for p in (.7, .8, .9):
            strengths = [Controller(clock=lambda: 10).command(decision(scores=[p]*3, depths=[d]*3), 0)['motor_strength_3'][0]
                         for d in (3., 2., 1., .5)]
            self.assertEqual(strengths, sorted(strengths))
            self.assertEqual(strengths[-1], round(p*100))

    def test_nearest_object_and_missing_depth_are_independent_per_motor(self):
        d = decision(depths=(2., None, .5))
        d['haptic_observation']['objects'].append(dict(current_zone_3='LEFT', depth_m=.5, realistic=True))
        result = self.controller.command(d, 0)
        self.assertEqual(result['motor_strength_3'], [90, 0, 90])
        self.assertEqual(result['motor_valid_3'], [True, False, True])
        self.assertEqual(result['status'], 'partial')
        for depth in (float('nan'), float('inf'), -1, True):
            r = Controller(clock=lambda: 10).command(decision(depths=(depth, 1, 1)), 0)
            self.assertFalse(r['motor_valid_3'][0])
            self.assertEqual(r['motor_strength_3'][0], 0)

    def test_clear_suppressed_unconfirmed_and_memory_do_not_buzz(self):
        for kind in ('clear', 'suppressed', 'unconfirmed', 'remembered'):
            d = decision()
            if kind == 'clear': d['haptic_observation']['objects'] = []
            if kind == 'suppressed': d['realistic_detection'] = False
            for obj in d['haptic_observation']['objects']:
                if kind == 'unconfirmed': obj['realistic'] = False
                if kind == 'remembered': obj['remembered'] = True
            c = Controller(clock=lambda: 10)
            self.assertEqual(c.command(d, 0)['motor_strength_3'], [0]*3)

    def test_hysteresis_and_smoothing(self):
        self.controller.command(decision(scores=[.7]*3, depths=[.5]*3), 0)
        self.now += .1
        r = self.controller.command(decision(self.now, 2, [.55]*3, [.5]*3), 0)
        self.assertTrue(all(self.controller.active))
        self.assertGreater(r['motor_strength_3'][0], 55)
        self.assertLess(r['motor_strength_3'][0], 70)
        self.now += .1
        self.controller.command(decision(self.now, 3, [.4]*3, [.5]*3), 0)
        self.assertFalse(any(self.controller.active))
        self.now += .1
        self.controller.command(decision(self.now, 4, [.55]*3, [.5]*3), 0)
        self.assertFalse(any(self.controller.active))
        # A confidence below the activation threshold never starts a new cue.
        self.assertEqual(Controller(clock=lambda: 10).command(decision(scores=[.55]*3), 0)['motor_strength_3'], [0]*3)

    def test_duplicate_delayed_out_of_order_and_invalidated_commands(self):
        d = decision()
        first = self.controller.command(d, 0)
        self.now += .4
        repeated = self.controller.command(d, 0)
        self.assertEqual(first['command_id'], repeated['command_id'])
        self.assertEqual(first['motor_strength_3'], repeated['motor_strength_3'])
        self.assertLess(repeated['ttl_ms'], first['ttl_ms'])
        self.controller.command(d, 0, enabled=False)
        self.assertFalse(self.controller.command(d, 0)['valid'])
        self.now = 11
        newer = self.controller.command(decision(11, 2), 0)
        self.assertTrue(newer['valid'])
        self.assertFalse(self.controller.command(d, 0)['valid'])
        self.now = 14
        self.assertFalse(self.controller.command(decision(11, 3), 0)['valid'])
        self.assertFalse(self.controller.command(decision(15, 4), 0)['valid'])

    def test_old_generation_and_bad_scores_stop_immediately(self):
        self.controller.command(decision(), 0)
        for score in (float('nan'), float('inf'), True, -1, 1.1, '0.9'):
            r = self.controller.command(decision(scores=[score]*3), 0)
            self.assertEqual(r['motor_strength_3'], [0]*3)
            self.assertFalse(r['valid'])
        self.assertFalse(self.controller.command(decision(), 1)['valid'])

    def test_backend_switch_resets_confidence_hysteresis(self):
        self.controller.command(decision(scores=[.9]*3), 0)
        self.now += .1
        d = decision(self.now, 2, [.55]*3)
        d['backend'] = 'laya'
        self.assertEqual(self.controller.command(d, 0)['motor_strength_3'], [0]*3)

    def test_invalid_settings(self):
        for kwargs in ({'far_m': .1}, {'activate': .3}, {'rise_s': 0}, {'max_age_s': 3}, {'near_m': float('nan')}):
            with self.assertRaises(ValueError): Settings(**kwargs)

    def test_decision_preserves_observation_age_through_slow_inference(self):
        forecaster = DecisionForecaster(backend='kinematic')
        scenario = {'realistic': True, 'confirmed_tracks': 1, 'motion_trend': 'left_obstacle',
                    'haptic_observation': decision()['haptic_observation']}
        result = forecaster.evaluate_sync(scenario)
        self.assertEqual(result['haptic_observation']['started'], 10)
        self.now = 13
        self.assertFalse(self.controller.command(result, 0)['valid'])

    def test_published_depth_and_decision_reach_command_with_generation_checks(self):
        state = State('live', Path('/tmp'), decision_backend='kinematic', depth_mode='off')
        now = time.monotonic()
        objects = [dict(current_zone_3='LEFT', depth_m=1.5, realistic=True, remembered=False)]
        scenario = dict(realistic=True, confirmed_tracks=1, motion_trend='left_obstacle',
                        predicted_zones_1s=[0]*6, predicted_zones_2s=[0]*6,
                        predicted_zones_3s=[0]*6, predicted_zones_3_2s=[0]*3)
        with patch.object(state.predictor, 'update', return_value=(objects, scenario)):
            state.publish(b'jpeg', {'objects': [], 'zones': [1,0,0,0,0,0], 'source_mode': 'live'}, now, 0)
        state.decision_forecaster.evaluate_sync(scenario)
        output = state.haptic_command()
        self.assertTrue(output['valid'])
        self.assertEqual(output['motor_strength_3'], [60, 0, 0])
        state.control(True)
        self.assertFalse(state.haptic_command()['valid'])
        state.control(False)
        state.frame, state.started = {'objects': []}, time.monotonic()
        self.assertFalse(state.haptic_command()['valid'])

    def test_simulation_preview_never_drives_hardware(self):
        state = State('simulation', Path('/tmp'), decision_backend='kinematic')
        now = time.monotonic()
        state.frame, state.started = {'objects': [], 'zones': [0]*6}, now
        state.decision_forecaster._latest_decision = decision(now)
        self.assertGreater(state.snapshot()['haptics']['motor_strength_3'][0], 0)
        self.assertEqual(state.haptic_command()['motor_strength_3'], [0]*3)
        self.assertFalse(state.haptic_command()['valid'])
        self.assertGreater(state.snapshot()['haptics']['motor_strength_3'][0], 0)

    def test_live_mode_cannot_send_synthetic_or_unidentified_observations(self):
        for source in ('simulation', 'replay', 'unknown'):
            state = State('live', Path('/tmp'), decision_backend='kinematic')
            now = time.monotonic()
            state.frame, state.started = {'objects': [], 'zones': [0]*6}, now
            d = decision(now)
            d['haptic_observation']['source_mode'] = source
            state.decision_forecaster._latest_decision = d
            self.assertFalse(state.haptic_command()['valid'])
            self.assertEqual(state.haptic_command()['motor_strength_3'], [0]*3)

    def test_command_listener_auth_and_route_isolation(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(lambda: self.controller.command(decision(), 0), 'test-token'))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f'http://127.0.0.1:{server.server_port}'
        try:
            with self.assertRaises(urllib.error.HTTPError) as err:
                urllib.request.urlopen(base+'/command')
            self.assertEqual(err.exception.code, 401)
            err.exception.close()
            request = urllib.request.Request(base+'/command', headers={'Authorization': 'Bearer test-token'})
            with urllib.request.urlopen(request) as response:
                result = json.load(response)
            self.assertEqual(result['schema_version'], 2)
            self.assertEqual(result['motor_strength_3'], [54,90,0])
            for path in ('/api/state', '/latest.jpg', '/api/decision'):
                with self.assertRaises(urllib.error.HTTPError) as err:
                    urllib.request.urlopen(urllib.request.Request(base+path, headers={'Authorization': 'Bearer test-token'}))
                self.assertEqual(err.exception.code, 404)
                err.exception.close()
        finally:
            server.shutdown(); server.server_close(); thread.join()


if __name__ == '__main__':
    unittest.main()
