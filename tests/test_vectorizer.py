from rag_from_scratch_v1.processing import Vectorizer,Chunk


def test_doc_sends_input_type_document_query_sends_query(make_doc, make_chunk, fake_client):
    client = fake_client
    vectorizer = Vectorizer(client=client, model="fake")
    text = ["test 1", "test 2"]
    doc = make_doc(text)
    list_chunk = make_chunk(doc)
    _ = vectorizer.doc(list_chunk)
    query = "text 3"
    _ = vectorizer.query(text=query)
    assert client.calls[0]['input_type'] == "document"
    assert client.calls[1]['input_type'] == "query"

def test_doc_returns_one_vector_per_chunk(make_doc, make_chunk, fake_client):
    client = fake_client
    vectorizer = Vectorizer(client=client, model="fake")
    text = ["test 1", "test 2"]
    doc = make_doc(text)
    list_chunk = make_chunk(doc)
    doc_embed = vectorizer.doc(list_chunk)
    assert len(text) == len(doc_embed)

def test_each_hunk_paired_with_embedding(make_doc, make_chunk, fake_client):
    client = fake_client
    vectorizer = Vectorizer(client=client, model="fake")
    text = ["test 1", "test 2"]
    doc = make_doc(text)
    list_chunk = make_chunk(doc)
    doc_embed = vectorizer.doc(list_chunk)
    assert doc_embed[0].chunk.text == text[0]
    assert doc_embed[0].embedding[0] == 0

def test_embeddings_are_tuples(make_doc, make_chunk, fake_client):
    client = fake_client
    vectorizer = Vectorizer(client=client, model="fake")
    text = ["test 1", "test 2"]
    doc = make_doc(text)
    list_chunk = make_chunk(doc)
    doc_embed = vectorizer.doc(list_chunk)
    query = "text 3"
    query_embed = vectorizer.query(text=query)
    assert isinstance(doc_embed[0].embedding, tuple)
    assert isinstance(query_embed, tuple)