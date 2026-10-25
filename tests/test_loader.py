import pytest

from rag_from_scratch_v1.processing import load_document


def test_load_pdf_returns_all_pages_in_order(make_pdf):
    # 1. Set up the input
    path = make_pdf(["Alpha page", "Beta page", "Gamma page"])

    # 2. Call the function
    doc = load_document(path)

    # 3. Check the result
    assert len(doc.pages) == 3
    assert [page.number for page in doc.pages] == [1, 2, 3]
    assert "Alpha" in doc.pages[0].text
    assert "Gamma" in doc.pages[2].text


@pytest.mark.parametrize("input", ["test.pdf", "TEST.PDF"])

def test_doc_type_extension_no_impact_on_load(make_pdf, input):
    path = make_pdf(["first page", "second_page"], name = input)
    doc = load_document(path)
    assert len(doc.pages) == 2
    assert "first" in doc.pages[0].text
    assert "second" in doc.pages[1].text 

def test_document_name_correctly_populated(make_pdf):
    pdf_name = "first"
    path = make_pdf(["first page"], name = pdf_name+".pdf")

    doc = load_document(path)
    assert doc.doc_name == pdf_name

def test_same_file_same_hash(make_pdf):
    path = make_pdf(["first_page"])

    doc = load_document(path)
    first_hash = doc.doc_hash

    doc = load_document(path)
    second_hash = doc.doc_hash

    assert first_hash == second_hash

def test_changed_content_different_hash(make_pdf):
    path = make_pdf(["first_page"])
    doc = load_document(path)
    first_hash = doc.doc_hash

    path = make_pdf(["second_page"])
    doc = load_document(path)
    second_hash = doc.doc_hash

    assert first_hash != second_hash

def test_blank_page_blank_output(make_pdf):
    path = make_pdf([""])
    doc = load_document(path)
    assert doc.pages[0].text == ""

def test_non_pdf_error(make_pdf):
    path = make_pdf(["first_page"], name = "test.doc")
    with pytest.raises(ValueError):
        doc = load_document(path)

def test_file_missing_error(tmp_path):
    path = tmp_path / "dummy.pdf"
    with pytest.raises(FileNotFoundError):
        doc = load_document(str(path))