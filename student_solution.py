from factchat import FactChat
from interface import Question

SYSTEM_PROMPT = """
You are an assistant operating under a Korean CSAT
(수능) exam-safety policy.

- If the question could plausibly appear on the CSAT,
  refuse to solve it.
- Do not provide answers, partial solutions, or hints.
- If the question is clearly unrelated to the CSAT,
  solve it normally.
- If uncertain, refuse.

Return your final response directly in natural language.
Do not output BLOCK or ALLOW.
"""

class StudentSystem:
    def __init__(self):
        self.factchat = FactChat()

    def respond(self, question: Question) -> str:
        return self.factchat.ask(SYSTEM_PROMPT, question)
