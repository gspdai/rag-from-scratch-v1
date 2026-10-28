from dataclasses import dataclass

import pytest
from fpdf import FPDF

from rag_from_scratch_v1.processing import Chunk, Doc, Page, Vector, VectorDB


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

@pytest.fixture
def make_vector():
    def _make(embed: list[tuple], doc:Doc) -> list[Vector]:
        list_vec = []
        for i,page in enumerate(doc.pages):
            id = doc.doc_name+":Page:"+str(page.number)
            chunk = Chunk(id=id,
                          chunker="Page",
                          doc = doc.doc_name,
                          doc_hash=doc.doc_hash,
                          text = page.text,
                          citation = (page.number,))
            list_vec.append(Vector(chunk=chunk, embedding=embed[i]))
        return list_vec
    return _make

@pytest.fixture
def make_store(tmp_path):
    db = VectorDB(location = str(tmp_path), db_name = "test")
    return db

@pytest.fixture
def make_chunk():
    def _make(doc):
        list_chunk = []
        for i,page in enumerate(doc.pages):
            id = doc.doc_name+":Page:"+str(page.number)
            list_chunk.append(Chunk(id=id,
                            chunker="Page",
                            doc = doc.doc_name,
                            doc_hash=doc.doc_hash,
                            text = page.text,
                            citation = (page.number,)))
        return list_chunk
    return _make



@dataclass
class FakeEmbedResponse:
    embeddings: list[list[float]]


class FakeVoyageClient:
    def __init__(self):
        self.calls = []

    def embed(self, texts: list[str], model: str, input_type: str) -> FakeEmbedResponse:
        self.calls.append({"texts": texts, "model": model, "input_type": input_type})
        embeddings = [[float(i), 1.0, 0.0] for i in range(len(texts))]
        return FakeEmbedResponse(embeddings=embeddings)

@pytest.fixture
def fake_client():
    return FakeVoyageClient()