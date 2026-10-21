from rag_from_scratch_v1.processing import load_document

doc = load_document("C:/Users/UjjwalDalmia/AI/rag-from-scratch-v1/data/Motor_Insurance_Comprehensive.pdf")
print(doc.doc_name, len(doc.pages))
print(doc.pages[10].text)