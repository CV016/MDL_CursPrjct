"""
shared/schemas.py
=================
Pydantic v2 data models shared across all AI-DOC INTERACT services.

Keeping schemas in a single module prevents schema drift between services
that communicate over HTTP — every service imports from here rather than
maintaining its own copy.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class FileType(str, Enum):
    """Supported document MIME / extension types."""

    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"


class InferenceVariant(str, Enum):
    """A/B experiment variant identifier."""

    A = "A"
    B = "B"


# ---------------------------------------------------------------------------
# Document upload / parse schemas
# ---------------------------------------------------------------------------


class ChunkSchema(BaseModel):
    """A single token-safe text chunk produced by the parser service."""

    chunk_id: int = Field(..., description="Zero-based sequential index of this chunk.")
    text: str = Field(..., description="Extracted and cleaned text content.")
    token_count: int = Field(..., ge=1, le=512, description="Number of tokens in this chunk.")


class DocumentUploadRequest(BaseModel):
    """
    Metadata accompanying a multipart document upload.

    The actual file bytes are transmitted as multipart/form-data;
    this schema validates any accompanying JSON fields.
    """

    filename: str = Field(..., min_length=1, max_length=255)
    file_type: FileType

    @field_validator("filename")
    @classmethod
    def filename_must_have_extension(cls, v: str) -> str:
        if "." not in v:
            raise ValueError("filename must include a file extension")
        return v


class ParsedDocumentResponse(BaseModel):
    """Full structured response returned by POST /parse."""

    filename: str
    file_type: FileType
    page_count: int = Field(..., ge=0)
    num_chunks: int = Field(..., ge=0)
    chunks: list[ChunkSchema]


# ---------------------------------------------------------------------------
# Inference request / response schemas
# ---------------------------------------------------------------------------


class InferenceRequest(BaseModel):
    """
    Payload sent from the gateway to an inference worker (summarizer or
    question_gen).  Carries the parsed document chunks and routing metadata.
    """

    doc_id: UUID = Field(..., description="Unique document identifier assigned by the gateway.")
    variant: InferenceVariant = Field(
        default=InferenceVariant.A,
        description="A/B experiment variant; controls which model checkpoint is used.",
    )
    chunks: list[ChunkSchema] = Field(..., min_length=1)
    # Optional caller-supplied hint (e.g. desired summary length)
    hint: Optional[str] = Field(default=None, max_length=500)


class InferenceResponse(BaseModel):
    """Unified response returned by both summarizer and question_gen workers."""

    doc_id: UUID
    variant: InferenceVariant
    # Summarizer populates `summary`; question_gen populates `questions`
    summary: Optional[str] = None
    questions: Optional[list[str]] = None
    # Wall-clock latency in milliseconds for observability
    latency_ms: float = Field(..., ge=0)


# ---------------------------------------------------------------------------
# Feedback schema
# ---------------------------------------------------------------------------


class FeedbackRequest(BaseModel):
    """
    Thumbs-up / thumbs-down feedback submitted from the frontend.
    Stored in the `feedback` table and used for offline model evaluation.
    """

    doc_id: UUID
    variant: InferenceVariant
    # 1 = positive (thumbs-up), -1 = negative (thumbs-down)
    score: int = Field(..., ge=-1, le=1)
    # Free-text comment is optional
    comment: Optional[str] = Field(default=None, max_length=2000)


class FeedbackResponse(BaseModel):
    """Acknowledgement returned after persisting feedback."""

    feedback_id: UUID
    message: str = "Feedback recorded successfully."


# ---------------------------------------------------------------------------
# Auth schemas (also used by the auth service internally)
# ---------------------------------------------------------------------------


class UserRegisterRequest(BaseModel):
    """Payload for POST /auth/register."""

    username: str = Field(..., min_length=3, max_length=64)
    email: str = Field(..., min_length=5, max_length=254)
    password: str = Field(..., min_length=8, max_length=128)


class UserLoginRequest(BaseModel):
    """Payload for POST /auth/token (OAuth2-style form data wrapper)."""

    username: str
    password: str


class TokenResponse(BaseModel):
    """JWT bearer token returned on successful login."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., description="Token validity in seconds.")


class CurrentUserResponse(BaseModel):
    """Payload returned by GET /auth/me."""

    user_id: UUID
    username: str
    email: str
