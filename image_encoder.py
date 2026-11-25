"""Local EmbeddingGemma 2 encoder for raw question images."""

from pathlib import Path

import numpy as np


class RawImageEncoder:
    def __init__(self, directory: Path):
        if not (directory / "model.safetensors").is_file():
            raise FileNotFoundError(f"Missing EmbeddingGemma 2 weights in {directory}. See README.md.")


        import torch
        from transformers import AutoConfig, AutoModel, AutoProcessor

        self.torch = torch
        if torch.cuda.is_available():
            self.device = "cuda"
            bfloat16 = torch.cuda.is_bf16_supported()
        else:
            on_mps = torch.backends.mps.is_available()
            self.device = "mps" if on_mps else "cpu"
            bfloat16 = on_mps

        config = AutoConfig.from_pretrained(directory, local_files_only=True)
        config.audio_config = None
        self.processor = AutoProcessor.from_pretrained(directory, local_files_only=True)
        self.model = AutoModel.from_pretrained(
            directory, config=config, local_files_only=True,
            dtype=torch.bfloat16 if bfloat16 else torch.float32,
        ).to(self.device).eval()

    def encode(self, path: Path) -> np.ndarray:
        from PIL import Image, ImageOps

        torch = self.torch
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        inputs = self.processor(text="<|image|>", images=image, images_kwargs={"max_soft_tokens": 140},
                                return_tensors="pt").to(self.device)
        with torch.inference_mode():
            hidden = self.model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = ((hidden * mask).sum(dim=1, dtype=torch.float32) /
                      mask.sum(dim=1, dtype=torch.float32)).to(hidden.dtype)
            # Match the normalization used to fit the SVM.
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            vector = torch.nn.functional.normalize(pooled, dim=-1)
        return vector[0].float().cpu().numpy()
