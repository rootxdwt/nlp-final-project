"""Score questions with the fitted local detectors."""

import os
import re
import unicodedata
from functools import cached_property
from pathlib import Path

import numpy as np

from image_encoder import RawImageEncoder
from interface import Question
from text_encoder import E5Encoder


ROOT = Path(__file__).resolve().parent
CHOICE = re.compile(r"[①②③④⑤❶❷❸❹❺]|^[ \t]*(?:[1-5][.)]|\([1-5a-eA-E]\))(?=\s)", re.MULTILINE)
POINTS = re.compile(r"[\[［(（]\s*[1-5]\s*점\s*[\]］)）]")


def content_text(text: str) -> str:
    without_markers = POINTS.sub(" ", CHOICE.sub(" ", text))
    normalized = unicodedata.normalize("NFKC", without_markers).casefold()
    return " ".join(normalized.split())


def text_score(vector: np.ndarray, references: np.ndarray) -> np.floating:
    similarities = references @ vector
    return np.partition(similarities, -3)[-3:].mean()


def image_score(vector: np.ndarray, support: np.ndarray, dual: np.ndarray, gamma: float) -> np.floating:
    difference = support - vector
    squared_distance = np.einsum("ij,ij->i", difference, difference)
    return np.exp(-gamma * squared_distance) @ dual


class CSATGate:
    def __init__(self, detector_path: Path = ROOT / "models" / "gate.npz"):
        self.model_dir = Path(os.getenv("CSAT_MODEL_DIR", str(ROOT / ".models" / "checkpoints"))).expanduser()
        with np.load(detector_path, allow_pickle=False) as detector:
            self.text_references = detector["text_references"]
            self.text_threshold = float(detector["text_threshold"])
            self.image_support = detector["image_support"].astype(np.float64)
            self.image_dual = detector["image_dual"]
            self.image_gamma = float(detector["image_gamma"])
            self.image_threshold = float(detector["image_threshold"])

    @cached_property
    def text_encoder(self) -> E5Encoder:
        return E5Encoder(self.model_dir / "e5")

    @cached_property
    def image_encoder(self) -> RawImageEncoder:
        return RawImageEncoder(self.model_dir / "gemma")

    def should_refuse(self, question: Question) -> bool:
        if question.type == "text":
            vector = self.text_encoder.encode(content_text(question.text or ""))
            return bool(text_score(vector, self.text_references) > self.text_threshold)

        vector = self.image_encoder.encode(Path(question.image_path or ""))
        score = image_score(vector, self.image_support, self.image_dual, self.image_gamma)
        return bool(score > self.image_threshold)
