import io
import json
import unittest
from unittest.mock import patch

from software.compare_models import nemotron
from software.evaluate_camera import summarize


class EvaluationTests(unittest.TestCase):
    def test_sampling_settings_and_image_preserved(self):
        response = {'done': True, 'done_reason': 'stop',
                    'message': {'content': '[0,1,1,0,0,0]'}}
        with patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(response).encode())) as send:
            result = nemotron(b'image bytes', 'http://localhost:11434', 30, 6, .1, 42)
        payload = json.loads(send.call_args.args[0].data)
        self.assertEqual(payload['options'], {'temperature': .1, 'seed': 42, 'num_predict': 64})
        self.assertIs(payload['think'], False)
        self.assertEqual(payload['messages'][0]['images'], ['aW1hZ2UgYnl0ZXM='])
        self.assertEqual(result['zones'], [0,1,1,0,0,0])

    def test_repeatability_excludes_incomplete_images(self):
        rows = []
        for sha, values in [('stable', [[0]*6, [0]*6]),
                            ('unstable', [[0]*6, [1]*6]),
                            ('failed', [[0]*6, None])]:
            for zones in values:
                rows.append(dict(backend='yolo', sha256=sha, zones=zones,
                                 valid=zones is not None, processing_ms=10))
        summary = summarize(rows)['yolo']
        self.assertEqual(summary['requests'], 6)
        self.assertEqual(summary['valid'], 5)
        self.assertEqual(summary['repeated_images_without_errors'], 2)
        self.assertEqual(summary['stable_images'], 1)


if __name__ == '__main__':
    unittest.main()
