"""Optional convenience client for the course-authorized FactChat model.

Students may design any internal pipeline in student_solution.py. The model
restriction is a course rule; the instructor should also enforce it through
FactChat account permissions and evaluation-server policies.
"""

import base64
import os
from pathlib import Path
from typing import Any

from interface import Question

BASE_URL = "https://factchat-cloud.mindlogic.ai/v1/gateway"
MODEL_ID = "gpt-5.4-mini"  # Instructor must verify this exact ID is enabled.
MIME_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


class FactChat:
    def __init__(self, client: Any = None):
        if client is not None:
            self.client = client  # Enables offline unit tests.
            return

        api_key = os.environ.get("FACTCHAT_API_KEY")
        if not api_key:
            raise RuntimeError("FACTCHAT_API_KEY is missing. Set it as an environment variable.")
        from openai import OpenAI
        self.client = OpenAI(
            api_key=api_key,
            base_url=BASE_URL,
            timeout=120.0,
            max_retries=0,
        )

    def ask(self, system_prompt: str, question: Question, instruction: str = "") -> str:
        """Ask the authorized model about one text or image question.

        This method makes one API request. The course starter calls it once
        per input question; other local preprocessing is allowed.
        """
        if question.type == "text":
            user_content: Any = (
                f"{instruction}\n\n{question.text}" if instruction else question.text
            )
        elif question.type == "image":
            path = Path(question.image_path or "")
            media_type = MIME_TYPES[path.suffix.lower()]
            image_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
            user_content = [
                {"type": "text", "text": instruction or "Read the question in this image."},
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_b64}"}},
            ]
        else:
            raise ValueError("Unsupported question type.")

        result = self.client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        content = result.choices[0].message.content
        return content if isinstance(content, str) else ""
