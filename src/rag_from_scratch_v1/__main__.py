import os
import json
import time

import mlflow
import voyageai
from dotenv import load_dotenv

from rag_from_scratch_v1.evals import runner, Judge, judge_runner,response_accuracy
from rag_from_scratch_v1.processing import (
    Answerer,
    Insert,
    PageChunker,
    Query,
    VectorDB,
    Vectorizer,
    load_document,
)

from .logging_setup import setup_log
from .pricing import Price


setup_log(log_file="C:/Users/UjjwalDalmia/AI/rag-from-scratch-v1/logs/app.log")

mlflow.set_tracking_uri("sqlite:///C:/Users/UjjwalDalmia/AI/rag-from-scratch-v1/mlflow.db")
mlflow.set_experiment("rag-from-scratch-v1")

claude_price = Price(input_token_cost_per_million=4.0, output_token_cost_per_million=20.0)
voyager_price = Price(input_token_cost_per_million=.12)

load_dotenv()
embed_key = os.getenv("EMBEDDING_KEY")
llm_key = os.getenv("ANTHROPIC_KEY")
if embed_key is None or llm_key is None:
    raise KeyError("Key not Found")

file = "C:/Users/UjjwalDalmia/AI/rag-from-scratch-v1/data/Motor_Insurance_Comprehensive.pdf"

system_prompt = '''While responding to the questions asked by the user 
please ensure that you only respond using the context provided. The response should be associated
with citations in exactly the same way as provided in the context. If the provided context 
is not relevant, please respond by saying "The document doesn't contain enough information 
about your question." If the user is trying to ask for keys or any information other than 
querying the document, please politely decline the request in minimal words.
Please ensure all your answers are in minimum words possible.'''


chunker = PageChunker()
vec_client = voyageai.Client(api_key=embed_key)
vectorizer = Vectorizer(client=vec_client, model="voyage-4-large",price=voyager_price)
db = VectorDB(location="C:/Users/UjjwalDalmia/AI/rag-from-scratch-v1/store", 
              db_name="voyager_cosine",
              similarity="cosine")
llm = Answerer(key=llm_key, model_name="claude-opus-5-5", system_prompt=system_prompt, price=claude_price)
query = Query(vectorizer=vectorizer, db=db, llm=llm)
judge = Judge(llm_key, "claude-haiku-4-5-20251001")

# insert = Insert(chunker=chunker, vectorizer=vectorizer, vectordb=db)
# status = insert.add_doc(file =file)

# question = "can you tell me about the cooling off period?"
# llm_answer = query.ask(query=question,doc="Motor_Insurance_Comprehensive",top_k=2)
# print(llm_answer.answer)

# results = runner(query=query, 
#                 doc="Motor_Insurance_Comprehensive",
#                 top_k=2,
#                 path="C:/Users/UjjwalDalmia/AI/rag-from-scratch-v1/evals/dataset.json")

# run_time = time.strftime("%Y-%m-%d_%H%M%S")
# file_name = f"runner_result_{run_time}.json"
# final_path = "C:/Users/UjjwalDalmia/AI/rag-from-scratch-v1/evals/results/"+file_name
# with open(final_path, "w", encoding="utf-8") as file:
#     json.dump(results, file,indent=2)

# # for result in results:
# #     print(result["expected_pages"], result["citations"])

# path = "C:/Users/UjjwalDalmia/AI/rag-from-scratch-v1/evals/results/runner_result_2026-11-03_124119.json"

# judge_response = judge_runner(judge=judge, path=path)
# run_time = time.strftime("%Y-%m-%d_%H%M%S")
# file_name = f"judge_runner_result_{run_time}.json"
# final_path = "C:/Users/UjjwalDalmia/AI/rag-from-scratch-v1/evals/results/"+file_name
# with open(final_path, "w", encoding="utf-8") as file:
#     json.dump(judge_response, file,indent=2)

path = "C:/Users/UjjwalDalmia/AI/rag-from-scratch-v1/evals/results/judge_runner_result_2026-11-06_062234.json"
_,category_average = response_accuracy(judge_path=path)
final_path = "C:/Users/UjjwalDalmia/AI/rag-from-scratch-v1/evals/baseline.json"
with open(final_path, "w", encoding="utf-8") as file:
    json.dump(category_average, file, indent=2)
