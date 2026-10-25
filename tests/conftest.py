import pytest
from fpdf import FPDF
from rag_from_scratch_v1.processing import Page, Doc


@pytest.fixture
def make_pdf(tmp_path):
    def _make(page_texts: list[str], name: str = "test.pdf"):
        pdf = FPDF()
        pdf.set_font("Helvetica", size=12)
        for text in page_texts:
            pdf.add_page()
            if text:
                pdf.cell(text=text)
        path = tmp_path / name
        pdf.output(str(path))
        return str(path)

    return _make

@pytest.fixture
def make_doc():
    def _make(pages: list[str], doc_name : str = "test.pdf") -> Doc:
        list_page = []
        for i, page in enumerate(pages):
            list_page.append(Page(number=i+1, text=page))
        tuple_pages = tuple(list_page)
        doc = Doc(doc_name=doc_name, 
                  doc_hash="dummy_hash",
                  pages=tuple_pages)
        return doc
    return _make
