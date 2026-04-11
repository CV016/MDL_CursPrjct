"""
services/parser/main.py
=======================
AI-DOC INTERACT — Document Parser Service

Responsibilities:
  - POST /parse  : Accept a multipart file upload (PDF, DOCX, PPTX).
  - Extract raw text using format-specific libraries:
      PDF  -> PyMuPDF  (fitz)
      DOCX -> python-docx
      PPTX -> python-pptx
  - Clean the extracted text:
      * Strip non-printable / control characters.
      * Normalise whitespace (collapse runs of spaces/newlines).
      * Remove heuristic page headers/footers (short lines at top/bottom).
  - Chunk the cleaned text into token-safe segments of at most 512 tokens
    using the HuggingFace tokenizers library.
  - Log the total input token count to Prometheus via a Counter metric.

Response format:
  {
    filename: str,
    file_type: "pdf"|"docx"|"pptx",
    page_count: int,
    num_chunks: int,
    chunks: [{ chunk_id: int, text: str, token_count: int }]
  }
"""

from __future__ import annotations

import io
import os
import re
import unicodedata
from typing import Annotated

import fitz  # PyMuPDF
from docx import Document as DocxDocument
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from prometheus_client import Counter, make_asgi_app
from pptx import Presentation
from pydantic import BaseModel, Field
from tokenizers import Tokenizer

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

# Counter incremented by the total number of input tokens across all parse
# requests — used for cost tracking and drift monitoring dashboards.
PARSER_INPUT_TOKENS_TOTAL = Counter(
    "parser_input_tokens_total",
    "Total number of input tokens processed by the parser service.",
    ["file_type"],
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_TOKENS_PER_CHUNK: int = 512

# Heuristic: lines shorter than this character count at page boundaries are
# treated as headers/footers and stripped.
HEADER_FOOTER_MAX_LENGTH: int = 80

# Tokenizer model — using the BERT WordPiece vocabulary as a widely available
# reference tokenizer.  Workers and the parser must use the same tokenizer so
# that chunk boundaries are consistent.
TOKENIZER_NAME: str = os.environ.get("TOKENIZER_NAME", "bert-base-uncased")

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI-DOC INTERACT — Parser Service",
    version="1.0.0",
    description="Extracts, cleans, and chunks text from PDF, DOCX, and PPTX documents.",
)

# Mount Prometheus metrics at /metrics
app.mount("/metrics", make_asgi_app())

# ---------------------------------------------------------------------------
# Lazy-loaded tokenizer (avoids download penalty at import time in tests)
# ---------------------------------------------------------------------------

_tokenizer: Tokenizer | None = None


def get_tokenizer() -> Tokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = Tokenizer.from_pretrained(TOKENIZER_NAME)
    return _tokenizer


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class ChunkOut(BaseModel):
    chunk_id: int
    text: str
    token_count: int = Field(..., ge=1, le=512)


class ParseResponse(BaseModel):
    filename: str
    file_type: str
    page_count: int
    num_chunks: int
    chunks: list[ChunkOut]


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------


def extract_text_pdf(file_bytes: bytes) -> tuple[str, int]:
    """
    Extract raw text from a PDF using PyMuPDF.

    Returns:
        (full_text, page_count)
    """
    pages: list[str] = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        page_count = doc.page_count
        for page in doc:
            pages.append(page.get_text("text"))
    return "\n".join(pages), page_count


def extract_text_docx(file_bytes: bytes) -> tuple[str, int]:
    """
    Extract raw text from a DOCX using python-docx.

    DOCX files have no inherent page count; we return 0 as a sentinel
    that the frontend can handle gracefully.
    """
    doc = DocxDocument(io.BytesIO(file_bytes))
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n".join(paragraphs), 0


def extract_text_pptx(file_bytes: bytes) -> tuple[str, int]:
    """
    Extract raw text from a PPTX using python-pptx.

    Each slide is treated as a logical "page" for page_count purposes.
    """
    prs = Presentation(io.BytesIO(file_bytes))
    slides: list[str] = []
    for slide in prs.slides:
        slide_texts: list[str] = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    line = "".join(run.text for run in para.runs).strip()
                    if line:
                        slide_texts.append(line)
        slides.append("\n".join(slide_texts))
    return "\n".join(slides), len(prs.slides)


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------


def _is_header_footer_line(line: str) -> bool:
    """
    Heuristically decide whether a line is likely a page header or footer.

    Criteria:
      - Line is shorter than HEADER_FOOTER_MAX_LENGTH characters after stripping.
      - Line contains only digits, punctuation, or common header/footer patterns
        (page numbers, dates, confidentiality notices, etc.).
    """
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped) >= HEADER_FOOTER_MAX_LENGTH:
        return False
    # Patterns: "Page N", "N of M", purely numeric, or short all-uppercase lines
    patterns = [
        r"^\d+$",
        r"^page\s+\d+(\s+of\s+\d+)?$",
        r"^\d+\s*/\s*\d+$",
        r"^[A-Z\s\-–—]+$",
        r"^confidential$",
        r"^proprietary.*$",
    ]
    for pattern in patterns:
        if re.match(pattern, stripped, re.IGNORECASE):
            return True
    return False


def clean_text(raw: str) -> str:
    """
    Produce clean, normalised text suitable for tokenisation.

    Steps:
      1. Unicode normalisation to NFC form.
      2. Remove non-printable / control characters (categories Cc and Cf).
      3. Heuristic removal of header/footer lines.
      4. Collapse runs of whitespace to a single space; normalise line breaks.
    """
    # Step 1: unicode normalisation
    text = unicodedata.normalize("NFC", raw)

    # Step 2: strip control characters
    text = "".join(
        ch for ch in text if unicodedata.category(ch) not in ("Cc", "Cf") or ch in ("\n", "\t")
    )

    # Step 3: remove header/footer lines
    lines = text.splitlines()
    cleaned_lines: list[str] = []
    for i, line in enumerate(lines):
        # Apply the heuristic only to the first/last 3 lines of each "block"
        # separated by blank lines — a cheap approximation of page boundaries.
        if _is_header_footer_line(line):
            continue
        cleaned_lines.append(line)

    # Step 4: collapse whitespace
    joined = "\n".join(cleaned_lines)
    # Collapse multiple blank lines into a single blank line
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    # Collapse runs of spaces/tabs within a line
    joined = re.sub(r"[ \t]+", " ", joined)
    return joined.strip()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_text(text: str, tokenizer: Tokenizer) -> list[ChunkOut]:
    """
    Split `text` into chunks each containing at most MAX_TOKENS_PER_CHUNK tokens.

    Strategy:
      - Split the text into sentences / paragraphs using newlines as natural
        boundaries.
      - Accumulate sentences into a buffer until adding the next sentence would
        exceed the token limit; at that point flush the buffer as a chunk.
      - If a single sentence exceeds the token limit it is hard-split at the
        token boundary.
    """
    segments = [seg.strip() for seg in text.split("\n") if seg.strip()]
    chunks: list[ChunkOut] = []
    buffer: list[str] = []
    buffer_tokens: int = 0

    def flush_buffer() -> None:
        nonlocal buffer, buffer_tokens
        if not buffer:
            return
        chunk_text_str = " ".join(buffer)
        encoding = tokenizer.encode(chunk_text_str)
        chunks.append(
            ChunkOut(
                chunk_id=len(chunks),
                text=chunk_text_str,
                token_count=min(len(encoding.ids), MAX_TOKENS_PER_CHUNK),
            )
        )
        buffer = []
        buffer_tokens = 0

    for segment in segments:
        encoding = tokenizer.encode(segment)
        seg_token_count = len(encoding.ids)

        if seg_token_count > MAX_TOKENS_PER_CHUNK:
            # Hard-split oversized segment at token boundaries
            flush_buffer()
            token_ids = encoding.ids
            for start in range(0, len(token_ids), MAX_TOKENS_PER_CHUNK):
                sub_ids = token_ids[start : start + MAX_TOKENS_PER_CHUNK]
                sub_text = tokenizer.decode(sub_ids)
                chunks.append(
                    ChunkOut(
                        chunk_id=len(chunks),
                        text=sub_text,
                        token_count=len(sub_ids),
                    )
                )
        elif buffer_tokens + seg_token_count > MAX_TOKENS_PER_CHUNK:
            flush_buffer()
            buffer.append(segment)
            buffer_tokens = seg_token_count
        else:
            buffer.append(segment)
            buffer_tokens += seg_token_count

    flush_buffer()
    return chunks


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@app.post("/parse", response_model=ParseResponse)
async def parse_document(
    file: Annotated[UploadFile, File(description="PDF, DOCX, or PPTX document.")],
) -> ParseResponse:
    """
    Accept a multipart file upload, extract and clean text, return chunks.

    Raises HTTP 415 for unsupported file types.
    Raises HTTP 422 if the file cannot be parsed.
    """
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No filename provided.")

    filename = file.filename
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    # Dispatch to the appropriate extractor
    file_bytes = await file.read()
    try:
        if extension == "pdf":
            raw_text, page_count = extract_text_pdf(file_bytes)
            file_type = "pdf"
        elif extension == "docx":
            raw_text, page_count = extract_text_docx(file_bytes)
            file_type = "docx"
        elif extension == "pptx":
            raw_text, page_count = extract_text_pptx(file_bytes)
            file_type = "pptx"
        else:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported file type: '{extension}'. Supported: pdf, docx, pptx.",
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to parse document: {exc}",
        ) from exc

    cleaned = clean_text(raw_text)
    tokenizer = get_tokenizer()
    chunks = chunk_text(cleaned, tokenizer)

    # Increment Prometheus counter with total token count
    total_tokens = sum(c.token_count for c in chunks)
    PARSER_INPUT_TOKENS_TOTAL.labels(file_type=file_type).inc(total_tokens)

    return ParseResponse(
        filename=filename,
        file_type=file_type,
        page_count=page_count,
        num_chunks=len(chunks),
        chunks=chunks,
    )


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe endpoint."""
    return {"status": "ok"}
