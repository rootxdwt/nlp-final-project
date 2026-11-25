"""Build the local gateway artifact from CSAT questions and held-out calibration data."""

import argparse
from collections.abc import Iterable
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from csat_gate import ROOT, content_text, image_score, text_score
from image_encoder import RawImageEncoder
from interface import Question
from text_encoder import E5Encoder

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def read_jsonl(path: Path) -> Iterable[tuple[int, dict]]:
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            yield number, record


def text_splits(directory: Path, years: tuple[int, int]) -> dict[int, list[str]]:
    text_path = directory / "csat_text.jsonl"
    meta_path = directory / "csat_meta.jsonl"
    texts = {year: [] for year in years}
    # The supplied export matches metadata to questions by record order.
    rows = list(read_jsonl(text_path))
    metadata = list(read_jsonl(meta_path))
    if len(rows) != len(metadata):
        raise ValueError(f"{text_path} and {meta_path} must have the same number of records")
    for (number, row), (_, meta) in zip(rows, metadata):
        year = meta.get("year")
        if year in texts:
            text = row.get("text")
            if row.get("type") != "text" or not isinstance(text, str) or not text.strip():
                raise ValueError(f"{text_path}:{number}: expected a nonempty text question")
            texts[year].append(text)
    return texts


def image_splits(directory: Path, years: tuple[int, int]) -> dict[int, list[Path]]:
    images = {year: [] for year in years}
    seen = set()
    # Deduplicate repeated common questions within each year and across the split.
    for year in years:
        for path in sorted(directory.glob(f"{year}*")):
            if path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file():
                continue
            digest = sha256(path.read_bytes()).digest()
            if digest not in seen:
                seen.add(digest)
                images[year].append(path)
    return images


def load_non_csat(path: Path | None) -> tuple[list[str], list[Path]]:
    if path is None:
        return [], []

    texts: list[str] = []
    images: list[Path] = []
    for number, row in read_jsonl(path):
        try:
            question = Question.from_dict(row)
        except ValueError as exc:
            raise ValueError(f"{path}:{number}: {exc}") from exc
        if question.type == "text":
            texts.append(row["text"])
        else:
            images.append(Path(row["image_path"]))
    if not texts and not images:
        raise ValueError(f"{path}: no non-CSAT questions found")
    return texts, images


def calibrate(label: str, positives: np.ndarray, negatives: np.ndarray, args: argparse.Namespace) -> float:
    if not np.isfinite(positives).all() or not np.isfinite(negatives).all():
        raise ValueError(f"{label}: non-finite calibration scores")
    if negatives.size:
        # The runner refuses only for scores strictly above the cutoff.
        allowed = math.floor(args.max_false_refusal * len(negatives))
        cutoff = np.sort(negatives)[len(negatives) - allowed - 1]
        print(f"{label} calibration: {np.count_nonzero(negatives > cutoff)}/{len(negatives)} false refusals", flush=True)
    else:
        caught = math.ceil(args.target_recall * len(positives))
        boundary = np.sort(positives)[len(positives) - caught]
        cutoff = np.nextafter(boundary, positives.dtype.type(-np.inf))
        print(f"{label} calibration: no non-CSAT examples, so false refusals could not be measured", flush=True)
    print(f"{label} calibration: {np.count_nonzero(positives > cutoff)}/{len(positives)} CSAT refusals", flush=True)
    return float(cutoff)


def embed_images(encoder: RawImageEncoder, paths: list[Path], label: str) -> np.ndarray:
    vectors = []
    for count, path in enumerate(paths, 1):
        vectors.append(encoder.encode(path))
        if count % 50 == 0 or count == len(paths):
            print(f"{label}: embedded {count}/{len(paths)} images", flush=True)
    return np.stack(vectors)


def write_artifact(path: Path, **arrays: np.ndarray | float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as directory:
        temporary_path = Path(directory) / "gate.npz"
        np.savez_compressed(temporary_path, **arrays)
        temporary_path.replace(path)


def train(args: argparse.Namespace) -> None:
    from sklearn.svm import OneClassSVM

    years = (args.fit_year, args.calibration_year)
    fit_year, calibration_year = years
    texts = text_splits(args.text_dir, years)
    images = image_splits(args.image_dir, years)
    if len(texts[fit_year]) < 3 or not images[fit_year]:
        raise ValueError(f"Fit year {fit_year} needs at least three text questions and one image")
    if not texts[calibration_year] or not images[calibration_year]:
        raise ValueError(f"Calibration year {calibration_year} needs text and image questions")
    non_csat_text, non_csat_images = load_non_csat(args.non_csat)

    print(f"Embedding {len(texts[fit_year])} text references from {fit_year}", flush=True)
    text_encoder = E5Encoder(args.model_dir / "e5")
    references = np.stack([text_encoder.encode(content_text(text)) for text in texts[fit_year]])
    text_positives = np.asarray([
        text_score(text_encoder.encode(content_text(text)), references) for text in texts[calibration_year]
    ])
    text_negatives = np.asarray([
        text_score(text_encoder.encode(content_text(text)), references) for text in non_csat_text
    ])
    text_cutoff = calibrate("Text", text_positives, text_negatives, args)

    image_encoder = RawImageEncoder(args.model_dir / "gemma")
    fit_images = embed_images(image_encoder, images[fit_year], f"Image fit ({fit_year})")
    svm = OneClassSVM(kernel="rbf", nu=args.nu, gamma=args.gamma).fit(fit_images)
    support = svm.support_vectors_
    dual = svm.dual_coef_.ravel()
    # The runtime uses the unshifted kernel sum, so the cutoff uses the same score.
    calibration_images = embed_images(image_encoder, images[calibration_year], f"Image calibration ({calibration_year})")
    image_positives = np.asarray([
        image_score(vector, support, dual, args.gamma) for vector in calibration_images
    ])
    image_negatives = np.asarray([
        image_score(image_encoder.encode(path), support, dual, args.gamma) for path in non_csat_images
    ])
    image_cutoff = calibrate("Image", image_positives, image_negatives, args)

    write_artifact(
        args.output,
        text_references=references,
        text_threshold=text_cutoff,
        image_support=support.astype(np.float32),
        image_dual=dual,
        image_gamma=args.gamma,
        image_threshold=image_cutoff,
    )
    print(f"Saved {args.output} ({len(references)} text references, {len(support)} image support vectors)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    model_dir = Path(os.getenv("CSAT_MODEL_DIR", str(ROOT / ".models" / "checkpoints"))).expanduser()
    parser.add_argument("--text-dir", type=Path, default=ROOT.parent / "text", help="CSAT text and metadata directory")
    parser.add_argument("--image-dir", type=Path, default=ROOT.parent / "image", help="Directory with year-prefixed CSAT images")
    parser.add_argument("--model-dir", type=Path, default=model_dir, help="Directory containing the e5 and gemma checkpoints")
    parser.add_argument("--fit-year", type=int, default=2025, help="CSAT year used for fitting")
    parser.add_argument("--calibration-year", type=int, default=2026, help="Held-out CSAT year for setting cutoffs")
    parser.add_argument("--non-csat", type=Path, help="Optional non-CSAT calibration questions in runner JSONL format")
    parser.add_argument("--target-recall", type=float, default=0.95, help="Minimum calibration recall without non-CSAT examples")
    parser.add_argument("--max-false-refusal", type=float, default=0.05, help="Maximum calibration false refusal rate with non-CSAT examples")
    parser.add_argument("--nu", type=float, default=0.1, help="One-Class SVM nu")
    parser.add_argument("--gamma", type=float, default=8.67846829877481, help="RBF gamma from the shipped image detector")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "gate.retrained.npz", help="Destination for the fitted artifact")
    args = parser.parse_args()
    if args.fit_year == args.calibration_year:
        parser.error("fit and calibration years must differ")
    if not (0 < args.target_recall <= 1 and 0 <= args.max_false_refusal < 1):
        parser.error("target recall must be in (0, 1] and maximum false refusal in [0, 1)")
    if not (0 < args.nu <= 1 and math.isfinite(args.gamma) and args.gamma > 0):
        parser.error("nu must be in (0, 1] and gamma must be finite and positive")
    try:
        train(args)
    except ModuleNotFoundError as exc:
        if exc.name != "sklearn":
            raise
        parser.exit(1, "Install the training dependency with python -m pip install -r gateway_trainer/requirements.txt\n")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Training failed: {exc}\n")


if __name__ == "__main__":
    main()
