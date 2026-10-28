from rag_from_scratch_v1.processing import Insert, Vectorizer, PageChunker, Status


def test_status_added(fake_client, make_store,make_pdf):
    vectorizer = Vectorizer(fake_client, model="fake")
    vector_db = make_store
    chunker = PageChunker()
    inserter = Insert(chunker=chunker, vectorizer=vectorizer, vectordb=vector_db)
    doc = make_pdf(["page 1", "page 2"])
    assert inserter.add_doc(doc) == Status.ADDED
    assert len(fake_client.calls) == 1

def test_status_existed(fake_client, make_store,make_pdf):
    vectorizer = Vectorizer(fake_client, model="fake")
    vector_db = make_store
    chunker = PageChunker()
    inserter = Insert(chunker=chunker, vectorizer=vectorizer, vectordb=vector_db)
    doc = make_pdf(["page 1", "page 2"])
    inserter.add_doc(doc)
    assert inserter.add_doc(doc) == Status.EXISTS
    assert len(fake_client.calls) == 1

def test_status_replaced(fake_client, make_store,make_pdf):
    vectorizer = Vectorizer(fake_client, model="fake")
    vector_db = make_store
    chunker = PageChunker()
    inserter = Insert(chunker=chunker, vectorizer=vectorizer, vectordb=vector_db)
    doc = make_pdf(["page 1", "page 2"])
    inserter.add_doc(doc)
    doc2 = make_pdf(["page 1", "page 3"])
    assert inserter.add_doc(doc2) == Status.REPLACED
    assert len(fake_client.calls) == 2
    stored = vector_db.collection.get()
    texts = " ".join(stored["documents"])
    assert vector_db.collection.count() == 2
    assert "page 3" in texts
    assert "page 2" not in texts