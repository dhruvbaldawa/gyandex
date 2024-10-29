from typing import Optional, Type
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
    ForeignKey,
    Text,
)
from sqlalchemy.orm import declarative_base, relationship, Session, sessionmaker
from sqlalchemy.sql import func

Base = declarative_base()


class Feed(Base):
    __tablename__ = "feeds"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text)
    author = Column(String(255))
    email = Column(String(255))
    image_url = Column(String(512))
    language = Column(String(50), default="en")
    copyright = Column(String(255))
    website = Column(String(512))
    categories = Column(String(512))  # Comma-separated list
    explicit = Column(String(10), default="no")
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())

    # Relationship
    episodes = relationship(
        "Episode", back_populates="feed", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Feed(name='{self.name}', title='{self.title}')>"


class Episode(Base):
    __tablename__ = "episodes"

    id = Column(Integer, primary_key=True)
    feed_id = Column(Integer, ForeignKey("feeds.id"), nullable=False)
    guid = Column(String(255), unique=True, nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text)
    audio_url = Column(String(512), nullable=False)
    publication_date = Column(DateTime, default=func.now())
    duration = Column(Integer)  # Duration in seconds
    episode_number = Column(Integer)
    season_number = Column(Integer)
    episode_type = Column(String(50), default="full")  # full, trailer, bonus
    explicit = Column(String(10), default="no")
    image_url = Column(String(512))
    file_size = Column(Integer)  # Size in bytes
    mime_type = Column(String(100))
    created_at = Column(DateTime, server_default=func.now())

    # Relationship
    feed = relationship("Feed", back_populates="episodes")

    def __repr__(self):
        return f"<Episode(title='{self.title}', feed='{self.feed.name}')>"


class PodcastDB:
    def __init__(self, db_path: str = "podcast.db"):
        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self.engine)
        self.session = sessionmaker(bind=self.engine)

    def create_feed(self, name: str, title: str, **kwargs) -> Feed:
        with self.session() as session:
            feed = Feed(name=name, title=title, **kwargs)
            session.add(feed)
            session.commit()
            session.refresh(feed)
            return feed

    def get_feed(self, name: str) -> Optional[Feed]:
        with self.session() as session:
            return session.query(Feed).filter(Feed.name == name).first()

    # @TODO: Update using the feed id, instead of name
    def add_episode(
        self, feed_name: str, title: str, audio_url: str, guid: str, **kwargs
    ) -> Episode:
        with self.session() as session:
            feed = session.query(Feed).filter(Feed.name == feed_name).first()
            if not feed:
                raise ValueError(f"Feed '{feed_name}' not found")

            episode = Episode(
                feed_id=feed.id, title=title, audio_url=audio_url, guid=guid, **kwargs
            )
            session.add(episode)
            session.commit()
            session.refresh(episode)
            return episode

    # @TODO: Update using the feed id, instead of name
    def get_episodes(self, feed_name: str, limit: int = None) -> list[Type[Episode]]:
        with self.session() as session:
            query = (
                session.query(Episode)
                .join(Feed)
                .filter(Feed.name == feed_name)
                .order_by(Episode.publication_date.desc())
            )

            if limit:
                query = query.limit(limit)

            return query.all()
