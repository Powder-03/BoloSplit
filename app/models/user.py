"""User and Friendship models."""

import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, Column
from sqlmodel import Field, Relationship, SQLModel


class User(SQLModel, table=True):
    """User account entity."""

    __tablename__ = "users"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
    firebase_uid: str | None = Field(default=None, index=True, max_length=128)
    email: str = Field(unique=True, index=True, max_length=255)
    name: str = Field(max_length=255)
    phone: str | None = Field(default=None, max_length=50)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        nullable=False,
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        nullable=False,
    )

    # Relationships
    friendships: list["Friendship"] = Relationship(
        sa_relationship_kwargs={
            "primaryjoin": "User.id==Friendship.user_id",
            "cascade": "all, delete-orphan",
        }
    )


class Friendship(SQLModel, table=True):
    """Friendship relationship with nickname and phonetic/spoken alias support."""

    __tablename__ = "friendships"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
    user_id: uuid.UUID = Field(foreign_key="users.id", index=True, nullable=False)
    friend_id: uuid.UUID = Field(foreign_key="users.id", index=True, nullable=False)
    nickname: str | None = Field(default=None, max_length=100)
    # List of spoken aliases like ["sahaj", "bhai", "roommate", "sahaj veer"]
    aliases: list[str] = Field(
        default_factory=list,
        sa_column=Column(JSON, nullable=False, default=list),
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        nullable=False,
    )
