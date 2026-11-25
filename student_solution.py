import logging

from csat_gate import CSATGate
from factchat import FactChat
from interface import Question

REFUSAL = "수능에 출제될 수 있는 문제라 풀이, 정답 또는 힌트를 제공할 수 없습니다."
UNAVAILABLE = "현재 질문에 답변할 수 없습니다."

SYSTEM_PROMPT = """You are a question-answering gateway. Solve the given problem.
"""

class StudentSystem:
    def __init__(self):
        self.gate = CSATGate()
        self.factchat = None

    def respond(self, question: Question) -> str:
        if self.gate.should_refuse(question):
            return "GATEREFUSE "+REFUSAL

        if self.factchat is None:
            self.factchat = FactChat()
        try:
            reply = self.factchat.ask(SYSTEM_PROMPT, question)
        except Exception as exc:
            logging.getLogger(__name__).warning("FactChat request failed (%s)", type(exc).__name__)
            return UNAVAILABLE

        return reply.strip()

        # decision, _, answer = reply.strip().partition("\n")
        # if decision.strip() == "ANSWER" and answer.strip():
        #     return answer.strip()
        # return REFUSAL
