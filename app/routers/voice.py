"""Voice command ingestion endpoints (Single-Pass Multimodal Direct-to-Ledger)."""

import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.database import get_session
from app.models.user import User
from app.schemas.voice import VoiceCommandRequest, VoiceIngestResponse
from app.services import ai_pipeline

router = APIRouter(prefix="/voice", tags=["Voice Ingestion"])


@router.post(
    "/text",
    response_model=VoiceIngestResponse,
    summary="Process voice command from text directly to ledger",
)
async def ingest_voice_text(
    payload: VoiceCommandRequest,
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> VoiceIngestResponse:
    """
    Ingest a text voice command (e.g., 'Add 500 for petrol split with Sahaj').

    Validates confidence and entity resolution strictly before mutating the ledger.
    """
    if not payload.text or not payload.text.strip():
        msg = "Sound was not clear. Please speak again clearly."
        return VoiceIngestResponse(
            status="ERROR",
            action_type="UNKNOWN",
            count=0,
            error_code="UNCLEAR_AUDIO",
            spoken_confirmation=msg,
            clarification_prompt=msg,
            is_ambiguous=False,
        )

    return await ai_pipeline.process_voice_pipeline(
        session=session,
        user=current_user,
        command_text=payload.text,
        client_timestamp_iso=payload.client_timestamp_iso,
        group_id=payload.group_id,
    )


@router.post(
    "/audio",
    response_model=VoiceIngestResponse,
    summary="Process raw audio recording in a single multimodal pass with directory context",
)
async def ingest_voice_audio(
    file: UploadFile = File(..., description="Audio recording file (e.g. wav, m4a, mp3, ogg, webm)"),
    client_timestamp_iso: str | None = Form(None),
    group_id: uuid.UUID | None = Form(None),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> VoiceIngestResponse:
    """
    Ingest raw audio directly from mobile client microphone in ONE multimodal Gemini call
    with full friend/group directory context.
    """
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded audio file is empty.",
        )

    mime_type = file.content_type or "audio/m4a"
    return await ai_pipeline.process_voice_pipeline(
        session=session,
        user=current_user,
        audio_bytes=audio_bytes,
        audio_mime_type=mime_type,
        client_timestamp_iso=client_timestamp_iso,
        group_id=group_id,
    )
