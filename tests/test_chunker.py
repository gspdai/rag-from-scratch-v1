from rag_from_scratch_v1.processing import PageChunker

def test_same_count_of_chunks_as_pages(make_doc):
    pdf_pages = ["first_page", "second_page"]
    doc = make_doc(pages = pdf_pages)
    chunks = PageChunker().chunk(doc)

    assert len(chunks) == len(pdf_pages)

def test_compare_chunk_text_with_pdf_text(make_doc):
    pdf_pages = ["first_page", "second_page"]
    doc = make_doc(pages = pdf_pages)
    chunks = PageChunker().chunk(doc)

    assert chunks[0].text == pdf_pages[0]
    assert chunks[1].text == pdf_pages[1]

def test_compare_chunk_id_format(make_doc):
    pdf_pages = ["first_page", "second_page"]
    doc = make_doc(pages = pdf_pages)
    chunks = PageChunker().chunk(doc)

    assert chunks[0].id == doc.doc_name+":page:1"

def test_blank_page_no_chunk(make_doc):
    pdf_pages = ["first_page","", "second_page"]
    doc = make_doc(pages = pdf_pages)
    chunks = PageChunker().chunk(doc)

    assert chunks[0].citation == (1,)
    assert chunks[1].citation == (3,)

def test_chunk_chunker_page(make_doc):
    pdf_pages = ["first_page","", "second_page"]
    doc = make_doc(pages = pdf_pages)
    chunks = PageChunker().chunk(doc)

    assert chunks[0].chunker == "page"
    assert chunks[1].chunker == "page"

def test_chunk_doc_same_as_doc_name(make_doc):
    pdf_pages = ["first_page","", "second_page"]
    doc = make_doc(pages = pdf_pages)
    chunks = PageChunker().chunk(doc)

    assert chunks[0].doc == doc.doc_name

def test_chunk_doc_hash_same_as_doc_hash(make_doc):
    pdf_pages = ["first_page","", "second_page"]
    doc = make_doc(pages = pdf_pages)
    chunks = PageChunker().chunk(doc)

    assert chunks[0].doc_hash == doc.doc_hash
    assert chunks[1].doc_hash == doc.doc_hash


def test_blank_page_empty_chunk_list(make_doc):
    pdf_pages = [""]
    doc = make_doc(pages = pdf_pages)
    chunks = PageChunker().chunk(doc)
    assert len(chunks) == 0
