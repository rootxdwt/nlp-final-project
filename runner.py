"""INSTRUCTOR-OWNED runner. Every team uses the exact same entry point.

Student contract:
    student_solution.StudentSystem.respond(Question) -> str

CLI examples:
    python runner.py --text '다음 중 옳은 것은? ...'
    python runner.py --image /path/to/question.png
    python runner.py --text-file /path/to/question.txt
    python runner.py --image /path/to/question.png --output answer.txt
    python runner.py --text "Question" --output answer.jsonl

Batch evaluation from a file (auto saves to results/):
    python runner.py --input examples/questions_4.jsonl
    python runner.py --input examples/questions_4.json --output my_results.jsonl

Batch evaluation (JSONL on stdin -> JSONL on stdout):
    python runner.py --jsonl < examples/inputs.jsonl
    python runner.py < examples/inputs.jsonl   # backwards-compatible default

In single-question CLI mode, stdout contains ONLY the natural-language answer.
In batch mode, stdout contains ONLY one {"response": "..."} JSON object per line.
Diagnostic logs belong on stderr, never stdout.
"""

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Protocol, Sequence, TextIO

from interface import Question


class RespondingSystem(Protocol):
    def respond(self, question: Question) -> str: ...


def checked_response(system: RespondingSystem, question: Question) -> str:
    """Call the student system without restricting its internal implementation."""
    result = system.respond(question)
    if not isinstance(result, str) or not result.strip():
        raise ValueError("respond() must return a nonempty natural-language string.")
    return result


def run_stream(system: RespondingSystem, source: TextIO, destination: TextIO) -> None:
    """Grade-friendly, newline-delimited JSON protocol for multiple questions."""
    for line_number, line in enumerate(source, 1):
        if not line.strip():
            continue
        try:
            question = Question.from_dict(json.loads(line))
            result = checked_response(system, question)
            output = {"response": result}
        except Exception as exc:
            # Emit a response for every input. The judge must mark an empty
            # response invalid; it is NOT a correct refusal.
            print(f"[line {line_number}] {type(exc).__name__}: {exc}", file=sys.stderr)
            output = {"response": ""}
        destination.write(json.dumps(output, ensure_ascii=False) + "\n")
        destination.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a student CSAT safety system on text or image questions."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--text", metavar="QUESTION", help="Question text (quote it in the shell).")
    group.add_argument("--text-file", metavar="PATH", help="UTF-8 text file containing a question.")
    group.add_argument("--image", metavar="PATH", help="PNG, JPEG, or WebP question image.")
    group.add_argument(
        "--jsonl", action="store_true",
        help="Batch mode: read one JSON question per stdin line, emit JSONL responses (default if no input flag).",
    )
    group.add_argument(
        "--input", metavar="PATH",
        help="Batch input file (.jsonl or .json); output saves automatically to results/ unless --output is given.",
    )
    parser.add_argument(
        "--output", metavar="PATH",
        help="Output file path. Without --input this is optional; default is stdout.",
    )
    return parser


def question_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Question | None:
    """Return None for batch mode; otherwise validate and build one Question."""
    if args.jsonl and any((args.text is not None, args.text_file is not None, args.image is not None)):
        parser.error("--jsonl cannot be combined with --text, --text-file, or --image.")
    try:
        if args.text is not None:
            return Question.from_dict({"type": "text", "text": args.text})
        if args.text_file is not None:
            text = Path(args.text_file).read_text(encoding="utf-8")
            return Question.from_dict({"type": "text", "text": text})
        if args.image is not None:
            return Question.from_dict({"type": "image", "image_path": args.image})
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))
    return None


def batch_file_lines(path: Path) -> TextIO:
    """Read JSONL directly, or turn a JSON array into JSONL for the fixed runner."""
    if path.suffix.lower() == ".jsonl":
        return path.open("r", encoding="utf-8")
    if path.suffix.lower() == ".json":
        questions = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(questions, list):
            raise ValueError("JSON input must contain an array of question objects.")
        return io.StringIO("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in questions))
    raise ValueError("--input must be a .jsonl or .json file.")


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    question = question_from_args(args, parser)
    if question is None and args.output and not args.input and not args.jsonl:
        parser.error("--output requires --input, --jsonl, --text, --text-file, or --image.")

    from student_solution import StudentSystem
    try:
        system = StudentSystem()
    except Exception as exc:
        print(f"[startup] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if question is None:
        # File input is the convenient no-shell-redirection mode.
        source_path = Path(args.input) if args.input is not None else None
        output_path = Path(args.output) if args.output else None
        if source_path is not None and output_path is None:
            output_path = Path("results") / f"{source_path.stem}_results.jsonl"
        try:
            with contextlib.ExitStack() as stack:
                source = stack.enter_context(batch_file_lines(source_path)) if source_path else sys.stdin
                if output_path is not None:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    destination = stack.enter_context(output_path.open("w", encoding="utf-8"))
                else:
                    destination = sys.stdout
                run_stream(system, source, destination)
            if output_path is not None:
                print(f"Saved: {output_path}", file=sys.stderr)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"[batch] {type(exc).__name__}: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
    else:
        try:
            response = checked_response(system, question)
            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                # The file suffix determines the serialization in single-question mode.
                # Unlike a .txt file, .jsonl must contain one valid JSON object per line.
                if output_path.suffix.lower() == ".jsonl":
                    data = json.dumps({"response": response}, ensure_ascii=False) + "\n"
                elif output_path.suffix.lower() == ".json":
                    data = json.dumps({"response": response}, ensure_ascii=False, indent=2) + "\n"
                else:
                    data = response + "\n"
                output_path.write_text(data, encoding="utf-8")
                print(f"Saved: {output_path}", file=sys.stderr)
            else:
                print(response)
        except Exception as exc:
            print(f"[question] {type(exc).__name__}: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
