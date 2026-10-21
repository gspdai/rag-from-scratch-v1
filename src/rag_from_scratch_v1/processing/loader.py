import os
import hashlib
from dataclasses import dataclass

from pypdf import PdfReader

SUPPORTED_FILETYPES = [".pdf"]

@dataclass(frozen=True)
class Page:
    number: int
    text: str

@dataclass(frozen=True)
class Doc:
    doc_name: str
    doc_hash:str
    pages: tuple[Page,...]

def load_document(file: str) -> Doc: 
    _file_name, file_type = os.path.splitext(file)
    _, file_name = os.path.split(_file_name)
    file_type = file_type.lower()

    if file_type not in SUPPORTED_FILETYPES:
        raise ValueError(f"file type {file_type} is not supported")

    with open(file, 'rb') as f:
        binary_data = f.read()
        doc_hash = hashlib.sha256(binary_data).hexdigest()

    if file_type == ".pdf":
        pages = _read_pdf(file)
        return Doc(doc_name= file_name, doc_hash=doc_hash, pages=pages)

def _read_pdf(file: str) -> tuple[Page,...]:
    reader = PdfReader(file)
    pages = []
    for i,read_page in enumerate(reader.pages, start=1):
        text = read_page.extract_text().strip()
        page = Page(number=i, text=text)
        pages.append(page)
    return tuple(pages)