"""Voice command ingestion endpoints."""

import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from google.genai import types
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.config import settings
from app.database import get_session
from app.models.user import User
from app.schemas.voice import VoiceCommandRequest, VoiceIngestResponse
from app.services import ai_pipeline

router = APIRouter(prefix="/voice", tags=["Voice Ingestion"])


@router.post(
    "/text",
    response_model=VoiceIngestResponse,
    summary="Process voice command from transcribed text into staging queue",
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
    summary="Process raw audio recording directly using Gemini Multimodal Perception",
)
async def ingest_voice_audio(
    file: UploadFile = File(..., description="Audio recording file (e.g. wav, m4a, mp3, ogg, webm)"),
    client_timestamp_iso: str | None = Form(None),
    group_id: uuid.UUID | None = Form(None),
    current_user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> VoiceIngestResponse:
    """
    Ingest raw audio directly from mobile client microphone.

    Uses Gemini 3.8 Flash multimodal audio perception to transcribe and extract intent.
    """
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded audio file is empty.",
        )

    client = ai_pipeline.get_genai_client()
    mime_type = file.content_type or "audio/m4a"

    # Multimodal audio transcription via Gemini
    try:
        response = await client.aio.models.generate_content(
            model=settings.STAGE1_MODEL,
            contents=[
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                (
                    "Transcribe this Hinglish/Indian English voice recording accurately. "
                    "If the audio is silent, too noisy, garbled, or unintelligible so you cannot "
                    "hear clear words, output only: [UNCLEAR]. Otherwise output only the verbatim transcript."
                ),
            ],
        )
        transcribed_text = response.text.strip() if response.text else ""
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audio perception failed: {exc}",
        ) from exc

    cleaned_marker = transcribed_text.strip("[](). ").upper()
    if not transcribed_text or "[UNCLEAR]" in transcribed_text.upper() or cleaned_marker in {
        "UNCLEAR",
        "INAUDIBLE",
        "NO_SPEECH",
        "SILENCE",
    }:
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
        command_text=transcribed_text,
        client_timestamp_iso=client_timestamp_iso,
        group_id=group_id,
    )
