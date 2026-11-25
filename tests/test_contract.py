import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from factchat import FactChat, MODEL_ID
from interface import Question
from runner import main, run_stream


class QuestionTests(unittest.TestCase):
    def test_text_input_contract(self):
        q = Question.from_dict({"type": "text", "text": "Question?"})
        self.assertEqual(q.text, "Question?")
        with self.assertRaises(ValueError):
            Question.from_dict({"type": "text", "text": "", "image_path": "x.png"})

    def test_image_input_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "item.png"
            image.write_bytes(b"\x89PNG\r\n\x1a\n")
            q = Question.from_dict({"type": "image", "image_path": str(image)})
            self.assertEqual(q.type, "image")

    def test_rejects_missing_image(self):
        with self.assertRaises(ValueError):
            Question.from_dict({"type": "image", "image_path": "/nonexistent/abc.png"})


class RunnerTests(unittest.TestCase):
    def test_same_runner_calls_system_for_each_question(self):
        class FakeStudent:
            def respond(self, question):
                return f"Seen {question.type}: {question.text or question.image_path}"

        incoming = io.StringIO(
            json.dumps({"type": "text", "text": "Hello"}) + "\n"
            + json.dumps({"type": "text", "text": "Goodbye"}) + "\n"
        )
        outgoing = io.StringIO()
        run_stream(FakeStudent(), incoming, outgoing)
        outputs = [json.loads(x) for x in outgoing.getvalue().splitlines()]
        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0], {"response": "Seen text: Hello"})
        self.assertEqual(outputs[1], {"response": "Seen text: Goodbye"})

    def test_errors_are_empty_and_must_be_marked_invalid(self):
        class BrokenStudent:
            def respond(self, question):
                raise RuntimeError("example error")

        incoming = io.StringIO('{"type":"text","text":"Hello"}\n')
        outgoing = io.StringIO()
        with patch("sys.stderr", io.StringIO()):
            run_stream(BrokenStudent(), incoming, outgoing)
        self.assertEqual(json.loads(outgoing.getvalue()), {"response": ""})


class CommandLineTests(unittest.TestCase):
    def _mock_student(self):
        class FakeStudent:
            def respond(self, question):
                if question.type == "text":
                    return f"TEXT: {question.text}"
                return f"IMAGE: {Path(question.image_path).name}"
        return FakeStudent()

    def test_cli_accepts_inline_text(self):
        output = io.StringIO()
        with patch("student_solution.StudentSystem", return_value=self._mock_student()), \
             patch("sys.stdout", output):
            main(["--text", "수능형 객관식 문제"])
        self.assertEqual(output.getvalue(), "TEXT: 수능형 객관식 문제\n")

    def test_cli_accepts_image_path(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "item.png"
            image.write_bytes(b"sample-image")
            output = io.StringIO()
            with patch("student_solution.StudentSystem", return_value=self._mock_student()), \
                 patch("sys.stdout", output):
                main(["--image", str(image)])
            self.assertEqual(output.getvalue(), "IMAGE: item.png\n")

    def test_cli_accepts_utf8_text_file(self):
        with tempfile.TemporaryDirectory() as directory:
            text_path = Path(directory) / "question.txt"
            text_path.write_text("첫 줄\n두 번째 줄", encoding="utf-8")
            output = io.StringIO()
            with patch("student_solution.StudentSystem", return_value=self._mock_student()), \
                 patch("sys.stdout", output):
                main(["--text-file", str(text_path)])
            self.assertEqual(output.getvalue(), "TEXT: 첫 줄\n두 번째 줄\n")

    def test_cli_batch_jsonl_still_works(self):
        output = io.StringIO()
        incoming = io.StringIO('{"type":"text","text":"Hello"}\n')
        with patch("student_solution.StudentSystem", return_value=self._mock_student()), \
             patch("sys.stdin", incoming), patch("sys.stdout", output):
            main(["--jsonl"])
        self.assertEqual(json.loads(output.getvalue()), {"response": "TEXT: Hello"})

    def test_batch_input_jsonl_autosaves_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "questions.jsonl"
            source.write_text('{"type":"text","text":"Hello"}\n', encoding="utf-8")
            with patch("student_solution.StudentSystem", return_value=self._mock_student()):
                import os
                old_cwd = os.getcwd()
                try:
                    os.chdir(root)
                    main(["--input", str(source)])
                finally:
                    os.chdir(old_cwd)
            saved = root / "results" / "questions_results.jsonl"
            self.assertTrue(saved.is_file())
            self.assertEqual(json.loads(saved.read_text(encoding="utf-8")), {"response": "TEXT: Hello"})

    def test_batch_input_json_array_with_explicit_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "questions.json"
            output = root / "out" / "answers.jsonl"
            source.write_text(json.dumps([
                {"type": "text", "text": "First"},
                {"type": "text", "text": "Second"},
            ]), encoding="utf-8")
            with patch("student_solution.StudentSystem", return_value=self._mock_student()):
                main(["--input", str(source), "--output", str(output)])
            actual = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(actual, [{"response": "TEXT: First"}, {"response": "TEXT: Second"}])

    def test_single_text_writes_output_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "answer.txt"
            with patch("student_solution.StudentSystem", return_value=self._mock_student()):
                main(["--text", "Hello", "--output", str(output)])
            self.assertEqual(output.read_text(encoding="utf-8"), "TEXT: Hello\n")

    def test_single_text_writes_valid_jsonl_when_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "answer.jsonl"
            with patch("student_solution.StudentSystem", return_value=self._mock_student()):
                main(["--text", "Hello", "--output", str(output)])
            lines = output.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0]), {"response": "TEXT: Hello"})

    def test_single_image_writes_valid_jsonl_when_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            img = root / "q.png"
            img.write_bytes(b"image")
            output = root / "answer.jsonl"
            with patch("student_solution.StudentSystem", return_value=self._mock_student()):
                main(["--image", str(img), "--output", str(output)])
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {"response": "IMAGE: q.png"})

    def test_output_without_input_is_rejected_instead_of_hanging(self):
        with patch("sys.stderr", io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                main(["--output", "results.jsonl"])
        self.assertEqual(raised.exception.code, 2)

    def test_cli_rejects_conflicting_modes(self):
        with patch("sys.stderr", io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                main(["--jsonl", "--text", "Question"])
        self.assertEqual(raised.exception.code, 2)


class FactChatRequestTests(unittest.TestCase):
    def test_authorized_model_id(self):
        self.assertEqual(MODEL_ID, "gpt-5.4-mini")

    def _fake_client(self):
        class Create:
            def __init__(self):
                self.kwargs = None

            def create(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ALLOW"))])

        create = Create()
        return SimpleNamespace(chat=SimpleNamespace(completions=create)), create

    def test_text_message_and_model(self):
        client, capture = self._fake_client()
        output = FactChat(client=client).ask("Check", Question(type="text", text="Sample"))
        self.assertEqual(output, "ALLOW")
        self.assertEqual(capture.kwargs["model"], MODEL_ID)
        self.assertEqual(capture.kwargs["messages"][1]["content"], "Sample")

    def test_image_passes_data_url(self):
        client, capture = self._fake_client()
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "sample.png"
            p.write_bytes(b"sample-image")
            FactChat(client=client).ask("Check", Question(type="image", image_path=str(p)))
        content = capture.kwargs["messages"][1]["content"]
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))


class StudentSystemTests(unittest.TestCase):
    def test_single_call_for_text_question(self):
        import student_solution
        with patch.object(student_solution, "FactChat") as maker:
            fake = maker.return_value
            fake.ask.return_value = "Final natural-language response"
            system = student_solution.StudentSystem()
            q = Question(type="text", text="sample")
            response = system.respond(q)
            self.assertEqual(response, "Final natural-language response")
            fake.ask.assert_called_once_with(student_solution.SYSTEM_PROMPT, q)

    def test_single_call_for_image_question(self):
        import student_solution
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "sample.jpg"
            p.write_bytes(b"sample-image")
            q = Question.from_dict({"type": "image", "image_path": str(p)})
            with patch.object(student_solution, "FactChat") as maker:
                fake = maker.return_value
                fake.ask.return_value = "Refused to solve"
                system = student_solution.StudentSystem()
                self.assertEqual(system.respond(q), "Refused to solve")
                fake.ask.assert_called_once_with(student_solution.SYSTEM_PROMPT, q)

    def test_one_call_per_question_in_batch(self):
        import student_solution
        with patch.object(student_solution, "FactChat") as maker:
            fake = maker.return_value
            fake.ask.side_effect = ["Refusal", "Solution"]
            system = student_solution.StudentSystem()
            questions = [Question(type="text", text="Q1"), Question(type="text", text="Q2")]
            answers = [system.respond(q) for q in questions]
            self.assertEqual(answers, ["Refusal", "Solution"])
            self.assertEqual(fake.ask.call_count, 2)  # 1 per question, not 2-stage


if __name__ == "__main__":
    unittest.main()
