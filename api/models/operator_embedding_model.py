from datetime import datetime

from sqlalchemy import Integer, ForeignKey, String, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from api.core.database import Base

EMBEDDING_DIM = 256


class OperatorEmbedding(Base):
    __tablename__ = "operator_embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    operator_id: Mapped[int] = mapped_column(
        ForeignKey("operators.id", ondelete="CASCADE"), nullable=False
    )
    embedding: Mapped[list[float]] = mapped_column(
        Vector(EMBEDDING_DIM), nullable=False
    )
    source_filename: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    operator: Mapped["Operator"] = relationship("Operator", back_populates="embeddings")  # type: ignore
