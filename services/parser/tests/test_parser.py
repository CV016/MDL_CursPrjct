"""
services/parser/tests/test_parser.py
======================================
Integration tests for the parser service.

These tests use small fixture files (PDF, DOCX, PPTX) bundled in the
tests/fixtures/ directory.  They exercise the full extraction -> cleaning
-> chunking pipeline without requiring a running server.

Run with:
    pytest services/parser/tests/ -v
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import pytest

# Allow importing the service module directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Provide required environment variable defaults before import
os.environ.setdefault("TOKENIZER_NAME", "bert-base-uncased")

import main as parser_main  # noqa: E402

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helper: create a minimal in-memory PDF for testing
# ---------------------------------------------------------------------------


def _make_minimal_pdf(text: str = "Hello world this is a test document.") -> bytes:
    """
    Produce a minimal single-page PDF in memory using PyMuPDF.
    Avoids the need for a fixture file on disk for the most basic tests.
    """
    import fitz  # type: ignore[import]

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=12)
    return doc.tobytes()  # type: ignore[return-value]


def _make_minimal_docx(text: str = "Hello world this is a test document.") -> bytes:
    """Produce a minimal in-memory DOCX."""
    from docx import Document  # type: ignore[import]

    doc = Document()
    doc.add_paragraph(text)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _make_minimal_pptx(text: str = "Hello world this is a test document.") -> bytes:
    """Produce a minimal in-memory PPTX with one text slide."""
    from pptx import Presentation  # type: ignore[import]
    from pptx.util import Pt  # type: ignore[import]

    prs = Presentation()
    blank_layout = prs.slide_layouts[5]
    slide = prs.slides.add_slide(blank_layout)
    txBox = slide.shapes.add_textbox(0, 0, prs.slide_width, prs.slide_height)
    tf = txBox.text_frame
    tf.text = text
    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Text extraction tests
# ---------------------------------------------------------------------------


class TestExtractTextPDF:
    def test_extracts_text_from_valid_pdf(self) -> None:
        content = "This content should appear after extraction."
        pdf_bytes = _make_minimal_pdf(content)
        text, page_count = parser_main.extract_text_pdf(pdf_bytes)
        assert content in text
        assert page_count == 1

    def test_page_count_matches_actual_pages(self) -> None:
        import fitz  # type: ignore[import]

        doc = fitz.open()
        for i in range(3):
            page = doc.new_page()
            page.insert_text((72, 72), f"Page {i + 1}", fontsize=12)
        pdf_bytes = doc.tobytes()

        _, page_count = parser_main.extract_text_pdf(pdf_bytes)
        assert page_count == 3


class TestExtractTextDOCX:
    def test_extracts_paragraphs_from_docx(self) -> None:
        content = "Paragraph extracted from DOCX."
        docx_bytes = _make_minimal_docx(content)
        text, page_count = parser_main.extract_text_docx(docx_bytes)
        assert content in text
        # DOCX has no native page count
        assert page_count == 0

    def test_empty_paragraphs_are_skipped(self) -> None:
        from docx import Document  # type: ignore[import]

        doc = Document()
        doc.add_paragraph("")
        doc.add_paragraph("Non-empty paragraph.")
        doc.add_paragraph("   ")
        buf = io.BytesIO()
        doc.save(buf)
        text, _ = parser_main.extract_text_docx(buf.getvalue())
        assert "Non-empty paragraph." in text
        assert text.count("\n") < 3  # blank paragraphs should not produce blank lines


class TestExtractTextPPTX:
    def test_extracts_text_from_pptx(self) -> None:
        content = "Slide text extracted from PPTX."
        pptx_bytes = _make_minimal_pptx(content)
        text, page_count = parser_main.extract_text_pptx(pptx_bytes)
        assert content in text
        assert page_count == 1

    def test_page_count_equals_slide_count(self) -> None:
        from pptx import Presentation  # type: ignore[import]

        prs = Presentation()
        blank_layout = prs.slide_layouts[5]
        for _ in range(4):
            prs.slides.add_slide(blank_layout)
        buf = io.BytesIO()
        prs.save(buf)
        _, page_count = parser_main.extract_text_pptx(buf.getvalue())
        assert page_count == 4


# ---------------------------------------------------------------------------
# Text cleaning tests
# ---------------------------------------------------------------------------


class TestCleanText:
    def test_removes_control_characters(self) -> None:
        raw = "Hello\x00\x01\x02 world\x1f"
        cleaned = parser_main.clean_text(raw)
        assert "\x00" not in cleaned
        assert "\x01" not in cleaned
        assert "Hello" in cleaned
        assert "world" in cleaned

    def test_collapses_multiple_blank_lines(self) -> None:
        raw = "Line one\n\n\n\n\nLine two"
        cleaned = parser_main.clean_text(raw)
        assert "\n\n\n" not in cleaned

    def test_strips_leading_trailing_whitespace(self) -> None:
        raw = "   \n\n  actual content  \n\n   "
        cleaned = parser_main.clean_text(raw)
        assert cleaned == cleaned.strip()

    def test_header_footer_lines_removed(self) -> None:
        raw = "123\nActual document content here.\nPage 1 of 5"
        cleaned = parser_main.clean_text(raw)
        assert "Actual document content here." in cleaned
        # Pure numeric and "Page N of M" lines should be gone
        assert "123" not in cleaned
        assert "Page 1 of 5" not in cleaned


# ---------------------------------------------------------------------------
# Chunking tests
# ---------------------------------------------------------------------------


class TestChunkText:
    """Verify chunking logic without hitting the HuggingFace hub by using a
    lightweight pre-initialised tokenizer from the main module."""

    def _get_tok(self) -> parser_main.Tokenizer:
        return parser_main.get_tokenizer()

    def test_single_short_text_produces_one_chunk(self) -> None:
        text = "This is a short sentence."
        tok = self._get_tok()
        chunks = parser_main.chunk_text(text, tok)
        assert len(chunks) == 1
        assert chunks[0].chunk_id == 0
        assert chunks[0].token_count <= parser_main.MAX_TOKENS_PER_CHUNK

    def test_long_text_produces_multiple_chunks(self) -> None:
        # Repeat a short sentence until we exceed one chunk's token budget
        sentence = "The quick brown fox jumps over the lazy dog. "
        text = sentence * 60  # ~720 tokens with bert-base
        tok = self._get_tok()
        chunks = parser_main.chunk_text(text, tok)
        assert len(chunks) > 1

    def test_every_chunk_within_token_limit(self) -> None:
        sentence = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        text = sentence * 80
        tok = self._get_tok()
        chunks = parser_main.chunk_text(text, tok)
        for chunk in chunks:
            assert chunk.token_count <= parser_main.MAX_TOKENS_PER_CHUNK

    def test_chunk_ids_are_sequential(self) -> None:
        text = "\n".join([f"Paragraph number {i}." for i in range(20)])
        tok = self._get_tok()
        chunks = parser_main.chunk_text(text, tok)
        for expected_id, chunk in enumerate(chunks):
            assert chunk.chunk_id == expected_id
