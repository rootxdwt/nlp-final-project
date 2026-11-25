"""Instructor-owned input contract. Do not change this file in submissions."""
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Any


@dataclass(frozen=True)
class Question:
    """Exactly one text question or one image question.

    Text:  {"type": "text", "text": "..."}
    Image: {"type": "image", "image_path": "/path/to/question.png"}
    """
    type: Literal["text", "image"]
    text: str | None = None
    image_path: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Question":
        if not isinstance(data, dict):
            raise ValueError("The question must be a JSON object.")
        kind = data.get("type")
        if kind == "text":
            if set(data) != {"type", "text"}:
                raise ValueError('Text input must contain exactly "type" and "text".')
            value = data["text"]
            if not isinstance(value, str) or not value.strip():
                raise ValueError('"text" must be a nonempty string.')
            return cls(type="text", text=value)
        if kind == "image":
            if set(data) != {"type", "image_path"}:
                raise ValueError('Image input must contain exactly "type" and "image_path".')
            value = data["image_path"]
            if not isinstance(value, str) or not value.strip():
                raise ValueError('"image_path" must be a nonempty string.')
            path = Path(value)
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                raise ValueError('Image must be a PNG, JPEG, or WebP file.')
            if not path.is_file():
                raise ValueError(f"Image file does not exist: {path}")
            return cls(type="image", image_path=str(path))
        raise ValueError('"type" must be either "text" or "image".')
