from datetime import datetime

from sqlalchemy import Integer, String, DateTime, Boolean, text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import ENUM

from api.core.database import Base
from api.models.enums import ProcessingStatus


class Operator(Base):
    __tablename__ = "operators"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    external_id: Mapped[str | None] = mapped_column(Integer, unique=True, nullable=True, index=True)
    status: Mapped[ProcessingStatus] = mapped_column(
        ENUM(ProcessingStatus, name="processing_status"),
        nullable=False,
        default=ProcessingStatus.PENDING,
        server_default=ProcessingStatus.PENDING.value,
    )
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)
    is_active: Mapped[bool] = mapped_column(
        Boolean, default=True, server_default=text("true")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    conversations: Mapped[list["Conversation"]] = relationship("Conversation", back_populates="operator")  # type: ignore
    embeddings: Mapped[list["OperatorEmbedding"]] = relationship("OperatorEmbedding", back_populates="operator", cascade="all, delete-orphan")  # type: ignore
