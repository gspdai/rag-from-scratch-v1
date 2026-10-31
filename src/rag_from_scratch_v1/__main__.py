import os
import voyageai
from dotenv import load_dotenv

from .logging_setup import setup_log

from rag_from_scratch_v1.processing import load_document, PageChunker, Vectorizer, VectorDB, Insert, Answerer, Query


setup_log(log_file="C:/Users/UjjwalDalmia/AI/rag-from-scratch-v1/logs/app.log")

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
vectorizer = Vectorizer(client=vec_client, model="voyage-4-large")
db = VectorDB(location="C:/Users/UjjwalDalmia/AI/rag-from-scratch-v1/store", 
              db_name="voyager_cosine",
              similarity="cosine")
llm = Answerer(key=llm_key, model_name="claude-opus-5-5", system_prompt=system_prompt)
query = Query(vectorizer=vectorizer, db=db, llm=llm)

# insert = Insert(chunker=chunker, vectorizer=vectorizer, vectordb=db)
# status = insert.add_doc(file =file)

question = "can you tell me about the cooling off period?"
answer = query.ask(query=question,doc="Motor_Insurance_Comprehensive",top_k=2)
print(answer)