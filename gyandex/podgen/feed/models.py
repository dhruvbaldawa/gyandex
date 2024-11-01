from typing import Optional, Type, Tuple
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
    ForeignKey,
    Text,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.sql import func

Base = declarative_base()


class Feed(Base):
    __tablename__ = "feeds"

    id = Column(Integer, primary_key=True)
    slug = Column(String(255), unique=True, nullable=False)
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
        return f"<Feed(slug='{self.slug}', title='{self.title}')>"

    def get_latest_episode(self, session) -> Tuple[int, int]:
        """
        Get the next episode number for the feed.
        """
        # Query the database to get the maximum episode number for the feed
        max_episode_number = (
            session.query(func.max(Episode.episode_number))
            .filter(Episode.feed_id == self.id)
            .scalar()
        ) or 0

        max_season_number = (
            session.query(func.max(Episode.season_number))
            .filter(Episode.feed_id == self.id)
            .scalar()
        ) or 1
        return max_season_number, max_episode_number

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

    def create_feed(self, slug: str, title: str, **kwargs) -> Feed:
        with self.session() as session:
            feed = Feed(slug=slug, title=title, **kwargs)
            session.add(feed)
            session.commit()
            session.refresh(feed)
            return feed

    def get_feed(self, slug: str) -> Optional[Feed]:
        with self.session() as session:
            return session.query(Feed).filter(Feed.slug == slug).first()

    def add_episode(
        self, feed_slug: str, title: str, audio_url: str, guid: str, **kwargs
    ) -> Episode:
        with self.session() as session:
            feed = session.query(Feed).filter(Feed.slug == feed_slug).first()
            if not feed:
                raise ValueError(f"Feed '{feed_slug}' not found")
            season_number, episode_number = feed.get_latest_episode(session)
            episode = Episode(
                feed_id=feed.id,
                title=title,
                audio_url=audio_url,
                guid=guid,
                season_number=season_number,
                episode_number=episode_number + 1,
                **kwargs,
            )
            session.add(episode)
            session.commit()
            session.refresh(episode)
            return episode

    # @TODO: Update using the feed id, instead of name
    def get_episodes(self, feed_slug: str, limit: int = None) -> list[Type[Episode]]:
        with self.session() as session:
            query = (
                session.query(Episode)
                .join(Feed)
                .filter(Feed.slug == feed_slug)
                .order_by(Episode.publication_date.desc())
            )

            if limit:
                query = query.limit(limit)

            return query.all()
