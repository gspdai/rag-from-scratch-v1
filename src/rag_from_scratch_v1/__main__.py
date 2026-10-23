import os
from dotenv import load_dotenv

from rag_from_scratch_v1.processing import load_document, PageChunker, Vectorizer, VectorDB, Insert

load_dotenv()
embed_key = os.getenv("EMBEDDING_KEY")
if embed_key is None:
    raise KeyError("Key not Found")

file = "C:/Users/UjjwalDalmia/AI/rag-from-scratch-v1/data/Motor_Insurance_Comprehensive.pdf"

chunker = PageChunker()
vectorizer = Vectorizer(key=embed_key, model="voyage-4-large")
db = VectorDB(location="C:/Users/UjjwalDalmia/AI/rag-from-scratch-v1/store", 
              db_name="voyager_cosine",
              similarity="cosine")

insert = Insert(chunker=chunker, vectorizer=vectorizer, vectordb=db)
status = insert.add_doc(file =file)

print(status)