"""Group and GroupMember models."""

import secrets
import uuid
from datetime import UTC, datetime

from sqlmodel import Field, Relationship, SQLModel


def generate_invite_code() -> str:
    """Generate a 6-character uppercase alphanumeric invite code."""
    return secrets.token_hex(3).upper()


class Group(SQLModel, table=True):
    """Shared expense group (e.g., 'Flatmates', 'Goa Trip')."""

    __tablename__ = "groups"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
    name: str = Field(max_length=255, index=True)
    invite_code: str = Field(
        default_factory=generate_invite_code,
        unique=True,
        index=True,
        max_length=20,
    )
    created_by: uuid.UUID = Field(foreign_key="users.id", nullable=False)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        nullable=False,
    )

    # Relationships
    members: list["GroupMember"] = Relationship(
        sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )


class GroupMember(SQLModel, table=True):
    """Association between User and Group."""

    __tablename__ = "group_members"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
    group_id: uuid.UUID = Field(foreign_key="groups.id", index=True, nullable=False)
    user_id: uuid.UUID = Field(foreign_key="users.id", index=True, nullable=False)
    role: str = Field(default="member", max_length=50)  # admin, member
    joined_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        nullable=False,
    )
