import json

from rag_from_scratch_v1.processing import Query

from .judge import Judge


def runner(query:Query, doc:str, top_k:int, path:str):
    with open(path, encoding="utf-8") as file:
        dataset = json.load(file)
    eval_response = []
    for record in dataset:
        question = record['question']
        llm_answer = query.ask(query=question,doc=doc,top_k=top_k)
        citations = sorted({citation for match in llm_answer.top_match for citation in match.chunk.citation})
        chunks = [match.chunk.text for match in llm_answer.top_match]
        record["llm_response"] = llm_answer.answer
        record["citations"] = citations
        record["chunk_text"] = chunks
        eval_response.append(record)
    return eval_response

def judge_runner(judge: Judge, path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as file:
        questions = json.load(file)

    judge_response = []

    for question in questions:
        answer = question["llm_response"]
        chunks = question["chunk_text"]
        query = question["question"]
        expected_answer = question["expected_answer"]

        judge_faithfulness = judge.faithfulness(answer=answer, chunks=chunks)
        judge_relevance = judge.relevance(question=query, answer=answer)
        judge_correctness = judge.correctness(expected_answer=expected_answer, answer=answer)

        question["faithfulness"] = {"score":judge_faithfulness.score, "reason":judge_faithfulness.reason}
        question["relevance"] = {"score":judge_relevance.score, "reason":judge_relevance.reason}
        question["correctness"] = {"score":judge_correctness.score, "reason":judge_correctness.reason}
        judge_response.append(question)

    return judge_response