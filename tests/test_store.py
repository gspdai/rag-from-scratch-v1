#from rag_from_scratch_v1.processing import VectorDB


def test_number_of_vectors_added_same(make_doc,make_vector,make_store):
    doc = make_doc(["first_page", "second_page"])
    vec = make_vector([(0,1,0), (0,0,1)], doc)
    make_store.add_doc(vec)
    assert len(vec) == make_store.collection.count()

def test_adding_empty_list_stores_nothing(make_doc, make_vector, make_store):
    make_store.add_doc([])
    assert make_store.collection.count() == 0

def test_doc_deleted_no_chunks(make_doc, make_vector, make_store):
    doc = make_doc(["first_page", "second_page"])
    vec = make_vector([(0,1,0), (0,0,1)], doc)
    make_store.add_doc(vec)
    make_store.del_doc("test.pdf")
    assert make_store.collection.count() == 0

def test_delete_one_doc_second_remains_intact(make_doc, make_vector, make_store):
    doc1 = make_doc(["first_page", "second_page"])
    vec1 = make_vector([(0,1,0), (0,0,1)], doc1)
    make_store.add_doc(vec1)
    doc2 = make_doc(["first_page", "second_page"], "test2.pdf")
    vec2 = make_vector([(0,1,0), (0,0,1)], doc2)
    make_store.add_doc(vec2)
    make_store.del_doc("test.pdf")
    assert make_store.collection.count() == 2

def test_doc_not_added_returns_false(make_store):
    assert make_store.check_exists("find.pdf") == False

def test_doc_added_returns_hash(make_doc, make_vector, make_store):
    doc1 = make_doc(["first_page", "second_page"])
    vec1 = make_vector([(0,1,0), (0,0,1)], doc1)
    make_store.add_doc(vec1)
    assert make_store.check_exists("test.pdf") == "dummy_hash"

def test_returns_top_k_matches_only(make_doc, make_vector, make_store):
    doc1 = make_doc(["first_page", "second_page", "third_page"])
    vec1 = make_vector([(0,1,0), (0,0,1), (1,1,1)], doc1)
    make_store.add_doc(vec1)
    assert len(make_store.search("test.pdf", (0,1,0), top_k = 2)) == 2

def test_chunks_from_requested_doc_only(make_doc, make_vector, make_store):
    doc1 = make_doc(["first_page", "second_page", "third_page"])
    vec1 = make_vector([(0,1,0), (0,0,1), (1,1,1)], doc1)
    make_store.add_doc(vec1)
    doc2 = make_doc(["first_page", "second_page"], "test2.pdf")
    vec2 = make_vector([(0,1,0), (0,0,1), (1,1,1)], doc2)
    make_store.add_doc(vec2)
    top_match = make_store.search("test.pdf", (0,1,0), top_k = 1)
    assert top_match[0].chunk.doc == "test.pdf"

def test_closest_vector_comes_first(make_doc, make_vector, make_store):
    doc1 = make_doc(["first_page", "second_page", "third_page"])
    vec1 = make_vector([(0,1,0), (0,0,1), (1,1,1)], doc1)
    make_store.add_doc(vec1)
    top_match = make_store.search("test.pdf", (0,1,0), top_k = 2)
    assert top_match[0].similarity > top_match[1].similarity


def test_top_k_more_than_doc_size(make_doc, make_vector, make_store):
    doc1 = make_doc(["first_page", "second_page", "third_page"])
    vec1 = make_vector([(0,1,0), (0,0,1), (1,1,1)], doc1)
    make_store.add_doc(vec1)
    assert len(make_store.search("test.pdf", (0,1,0), top_k = 4)) == 3
