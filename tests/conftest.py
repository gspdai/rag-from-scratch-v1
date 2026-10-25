import pytest
from fpdf import FPDF


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
