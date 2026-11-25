"""Authentication dependency supporting Firebase Auth (Google & Phone OTP) with dev-only mock fallback."""

import json
import logging
import os
from pathlib import Path

import firebase_admin
from fastapi import Depends, Header, HTTPException, status
from firebase_admin import auth as fb_auth
from firebase_admin import credentials
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_session
from app.models.user import User

_firebase_initialized = False


def init_firebase() -> bool:
    """Initialize Firebase Admin SDK once if credentials or project ID are provided."""
    global _firebase_initialized
    if _firebase_initialized or firebase_admin._apps:
        _firebase_initialized = True
        return True

    if settings.FIREBASE_CREDENTIALS_JSON:
        try:
            cert_dict = json.loads(settings.FIREBASE_CREDENTIALS_JSON)
            cred = credentials.Certificate(cert_dict)
            firebase_admin.initialize_app(cred)
            _firebase_initialized = True
            return True
        except Exception as e:
            logging.getLogger(__name__).warning("Failed to initialize Firebase from FIREBASE_CREDENTIALS_JSON: %s", e)

    if settings.FIREBASE_CREDENTIALS_BASE64:
        try:
            import base64

            decoded = base64.b64decode(settings.FIREBASE_CREDENTIALS_BASE64).decode("utf-8")
            cert_dict = json.loads(decoded)
            cred = credentials.Certificate(cert_dict)
            firebase_admin.initialize_app(cred)
            _firebase_initialized = True
            return True
        except Exception as e:
            logging.getLogger(__name__).warning("Failed to initialize Firebase from FIREBASE_CREDENTIALS_BASE64: %s", e)

    render_secret = Path("/etc/secrets/firebase-service-account.json")
    if render_secret.exists():
        cred = credentials.Certificate(str(render_secret))
        firebase_admin.initialize_app(cred)
        _firebase_initialized = True
        return True

    if settings.FIREBASE_CREDENTIALS_PATH and os.path.exists(settings.FIREBASE_CREDENTIALS_PATH):
        cred = credentials.Certificate(settings.FIREBASE_CREDENTIALS_PATH)
        firebase_admin.initialize_app(cred)
        _firebase_initialized = True
        return True

    default_json = Path("./firebase-service-account.json")
    if default_json.exists():
        cred = credentials.Certificate(str(default_json))
        firebase_admin.initialize_app(cred)
        _firebase_initialized = True
        return True

    if settings.FIREBASE_PROJECT_ID:
        firebase_admin.initialize_app(options={"projectId": settings.FIREBASE_PROJECT_ID})
        _firebase_initialized = True
        return True

    return False


async def get_current_user(
    authorization: str | None = Header(
        default=None,
        description="Firebase Bearer ID token (e.g., 'Bearer eyJhbGci...'). Generated on client via Firebase Auth SDK.",
    ),
    x_user_email: str | None = Header(
        default=None,
        alias="X-User-Email",
        description="Mock auth header for local dev / testing when Firebase token is absent.",
    ),
    session: AsyncSession = Depends(get_session),
) -> User:
    """
    Authenticate user via Firebase ID Token (Bearer) or local development mock fallback.
    """
    if authorization and authorization.startswith("Bearer "):
        id_token = authorization.split(" ", 1)[1].strip()

        if not init_firebase():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Firebase is not configured on the backend. Provide firebase-service-account.json or set FIREBASE_CREDENTIALS_PATH.",
            )

        try:
            decoded_token = fb_auth.verify_id_token(id_token)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid or expired Firebase ID token: {exc}",
            ) from exc

        fb_uid = decoded_token.get("uid")
        fb_email = (decoded_token.get("email") or "").strip().lower() or None
        fb_phone = (decoded_token.get("phone_number") or "").strip() or None
        fb_name = decoded_token.get("name") or (fb_email.split("@")[0].title() if fb_email else "BoloSplit User")

        filters = [User.firebase_uid == fb_uid]
        if fb_email:
            filters.append(User.email == fb_email)
        if fb_phone:
            filters.append(User.phone == fb_phone)

        result = await session.execute(select(User).where(or_(*filters)))
        user = result.scalars().first()

        if user:
            needs_commit = False
            if not user.firebase_uid:
                user.firebase_uid = fb_uid
                needs_commit = True
            if fb_name and user.name.startswith("Friend "):
                user.name = fb_name
                needs_commit = True
            if fb_phone and not user.phone:
                user.phone = fb_phone
                needs_commit = True
            if needs_commit:
                session.add(user)
                await session.commit()
                await session.refresh(user)
            return user

        user_email = fb_email or f"user_{fb_uid[:10]}@bolosplit.app"
        user = User(
            firebase_uid=fb_uid,
            email=user_email,
            name=fb_name,
            phone=fb_phone,
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user

    # Restrict header/mock fallback strictly to non-production environments
    if settings.ENVIRONMENT.strip().lower() not in ("development", "dev", "test", "testing", "local"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required: Provide a valid Firebase Bearer token.",
        )

    email = x_user_email.strip() if x_user_email else settings.DEFAULT_USER_EMAIL
    if not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed: Provide Firebase Bearer token or X-User-Email header.",
        )

    result = await session.execute(select(User).where(User.email == email))
    user = result.scalars().first()

    if not user:
        user_name = settings.DEFAULT_USER_NAME if email == settings.DEFAULT_USER_EMAIL else email.split("@")[0].title()
        user = User(email=email, name=user_name)
        session.add(user)
        await session.commit()
        await session.refresh(user)

    return user
