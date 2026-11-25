"""Offline integration test for the provided four-image batch fixture."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from runner import main
from interface import Question

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'examples' / 'image_questions_4.jsonl'


class ImageBatchExamplesTests(unittest.TestCase):
    def test_all_four_images_exist_and_load(self):
        lines = INPUT.read_text(encoding='utf-8').splitlines()
        self.assertEqual(len(lines), 4)
        for line in lines:
            item = json.loads(line)
            image_path = ROOT / item['image_path']
            self.assertTrue(image_path.is_file(), f'Missing: {image_path}')
            Question.from_dict({'type': 'image', 'image_path': str(image_path)})

    def test_batch_produces_four_jsonl_responses(self):
        class FakeStudent:
            def respond(self, question):
                return 'Processed image: ' + Path(question.image_path).name

        # Since sample paths are relative to project root, run from that root.
        import os
        old = Path.cwd()
        try:
            os.chdir(ROOT)
            with tempfile.TemporaryDirectory() as td:
                output = Path(td) / 'batch_outputs.jsonl'
                with patch('student_solution.StudentSystem', return_value=FakeStudent()):
                    main(['--input', str(INPUT), '--output', str(output)])
                outputs = [json.loads(line) for line in output.read_text(encoding='utf-8').splitlines()]
                self.assertEqual(len(outputs), 4)
                self.assertTrue(all('Processed image:' in x['response'] for x in outputs))
        finally:
            os.chdir(old)


if __name__ == '__main__':
    unittest.main()
