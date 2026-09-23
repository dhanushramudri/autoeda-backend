from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from ..database import Base


def _now():
    return datetime.now(timezone.utc)


class CoePost(Base):
    """A short-lived community post — newsletters, upcoming events, technical
    findings/tips, certification opportunities, and general resources that
    otherwise get shared once in Slack/Teams/email and are forgotten within
    a day. One shared feed across the whole DS practice, same free-for-all
    convention as DocCategory/DocArticle (any authenticated user can post;
    not gatekept per-workspace).
    """
    __tablename__ = "coe_posts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # newsletter | event | finding | certification | resource
    category: Mapped[str] = mapped_column(String(20), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    link_url: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    # Registration deadline / event date / cert deadline — whatever makes
    # this post time-sensitive. Used to sort upcoming items to the top and
    # to visually flag something as expiring soon.
    event_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    tags_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
