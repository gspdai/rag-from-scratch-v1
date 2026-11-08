from anthropic import Anthropic
from rag_from_scratch_v1.processing import Response 
from dataclasses import dataclass


FAITHFULNESS_PROMPT = """You are an evaluation judge.
You will receive an ANSWER and the CHUNKS of document text it was supposed to be based on.
Decide whether every factual claim in the ANSWER is supported by the CHUNKS.
Score 1 if all claims are supported, or if the answer only says the document does not contain the information.
Score 0 if any claim is not supported by the CHUNKS, even if it might be true.
Do not judge whether the answer is correct or complete, only whether it is supported.
Give a one-line reason."""

RELEVANCE_PROMPT = """You are an evaluation judge.
You will receive a QUESTION and the ANSWER an AI assistant gave to it.
Decide whether the ANSWER directly responds to what the QUESTION asks.
Score 1 if the answer addresses the question that was asked, even if it is short.
Score 1 if the answer says the document does not contain the information, since that is a direct response to the question.
Score 0 if the answer avoids the question, answers a different question, or is mostly about unrelated topics.
Do not judge whether the answer is correct or supported, only whether it responds to the question.
Give a one-line reason."""

CORRECTNESS_PROMPT = """You are an evaluation judge.
You will receive an ANSWER from an AI assistant and the EXPECTED ANSWER written by a human expert.
Decide whether the ANSWER contains all the key facts in the EXPECTED ANSWER, with no facts that contradict it.
Compare meaning, not wording: different phrasing of the same fact counts as a match.
Extra correct detail beyond the EXPECTED ANSWER is fine and does not lower the score.
Score 1 if every key fact in the EXPECTED ANSWER is present and nothing contradicts it.
Score 0 if any key fact is missing, wrong, or contradicted.
If the EXPECTED ANSWER says the document does not contain the information, score 1 only if the ANSWER also says the information is not available and does not invent an answer.
Give a one-line reason naming any missing or wrong fact."""


JUDGE_TOOL = {
     "name":"record_response",
     "description":"Record the binary score and reason for it",
     "input_schema":{
          "type":"object",
          "properties":{
               "score":{"type": "integer", "enum":[0,1]},
               "reason":{"type": "string"}
          },
          "required":["score", "reason"]
     }
     
}


@dataclass(frozen=True)
class Judge_Response:
     score:int
     reason:str


class Judge():
    def __init__(self, key:str, model_name:str):
        self.client = Anthropic(api_key = key)
        self.model = model_name
        self.max_token = 1024

    def _ask_judge(self, system_prompt:str, user_query:str)-> Judge_Response:
            message = self.client.messages.create(
                 max_tokens = self.max_token,
                 model = self.model,
                 system=system_prompt,
                 messages=[{"role": "user","content": user_query}],
                 tools = [JUDGE_TOOL],
                 tool_choice={"type": "tool", "name": "record_response"})
            for block in message.content:
                  if block.type == "tool_use":
                        return Judge_Response(score = block.input["score"], reason=block.input["reason"])
            raise ValueError("Judge did not return a judgement")

    def faithfulness(self, answer: str, chunks: list[str]) -> Judge_Response:
        chunk_text = "\n\n".join(chunks)
        content = f"<chunks>\n{chunk_text}\n</chunks>\n\n<answer>\n{answer}\n</answer>"
        return self._ask_judge(FAITHFULNESS_PROMPT, content)

    def relevance(self, question: str, answer: str) -> Judge_Response:
        content = f"<question>\n{question}\n</question>\n\n<answer>\n{answer}\n</answer>"
        return self._ask_judge(RELEVANCE_PROMPT, content)

    def correctness(self, expected_answer: str, answer: str) -> Judge_Response:
        content = f"<expected_answer>\n{expected_answer}\n</expected_answer>\n\n<answer>\n{answer}\n</answer>"
        return self._ask_judge(CORRECTNESS_PROMPT, content)