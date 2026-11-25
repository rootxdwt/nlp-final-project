"""Local multilingual E5 encoder using the CPU ONNX export."""

from pathlib import Path

import numpy as np


class E5Encoder:
    def __init__(self, directory: Path):
        tokenizer_path = directory / "tokenizer.json"
        model_path = directory / "model_qint8_avx512_vnni.onnx"
        if not tokenizer_path.is_file() or not model_path.is_file():
            raise FileNotFoundError(f"Incomplete E5 model directory: {directory}. See the README setup steps.")

        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length=512)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(
            str(model_path), options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {item.name for item in self.session.get_inputs()}

    def encode(self, text: str) -> np.ndarray:
        tokens = self.tokenizer.encode("query: " + text)
        inputs = {
            "input_ids": np.asarray([tokens.ids], dtype=np.int64),
            "attention_mask": np.asarray([tokens.attention_mask], dtype=np.int64),
            "token_type_ids": np.asarray([tokens.type_ids], dtype=np.int64),
        }
        hidden = self.session.run(None, {key: inputs[key] for key in self.input_names})[0]
        mask = inputs["attention_mask"][..., None]
        pooled = ((hidden * mask).sum(axis=1) / mask.sum(axis=1))[0]
        return (pooled / np.linalg.norm(pooled)).astype(np.float32)
