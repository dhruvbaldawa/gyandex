from typing import Optional, Dict, Any, List, Type
import os
import hashlib
from datetime import datetime
import mutagen
from dataclasses import dataclass
from urllib.parse import urljoin

from .feed import PodcastFeedGenerator
from .storage import S3CompatibleStorage
from .models import PodcastDB, Episode


@dataclass
class PodcastMetadata:
    title: str
    description: str
    duration: Optional[int] = None
    episode_number: Optional[int] = None
    season_number: Optional[int] = None
    episode_type: str = "full"
    explicit: str = "no"
    image_url: Optional[str] = None
    publication_date: Optional[datetime] = None


class PodcastOrchestrator:
    def __init__(
        self,
        storage: S3CompatibleStorage,
        db: PodcastDB,
        base_url: str,
        audio_prefix: str = "episodes",
        feed_prefix: str = "feeds",
    ):
        """
        Initialize the podcast orchestrator.

        Args:
            storage: Instance of S3CompatibleStorage
            db: Instance of PodcastDB
            base_url: Base URL for generating public URLs
            audio_prefix: Prefix for audio files in storage
            feed_prefix: Prefix for feed files in storage
        """
        self.storage = storage
        self.db = db
        self.base_url = base_url.rstrip("/")
        self.audio_prefix = audio_prefix.strip("/")
        self.feed_prefix = feed_prefix.strip("/")
        self.feed_generator = PodcastFeedGenerator(db)

    def _get_audio_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from audio file."""
        audio = mutagen.File(file_path)
        metadata = {}

        if audio is not None:
            metadata["duration"] = (
                int(audio.info.length) if hasattr(audio.info, "length") else None
            )
            metadata["mime_type"] = (
                audio.mime[0] if hasattr(audio, "mime") and audio.mime else None
            )

        metadata["file_size"] = os.path.getsize(file_path)
        return metadata

    def _generate_guid(self, feed_name: str, file_path: str) -> str:
        """Generate a unique GUID for the episode."""
        with open(file_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return f"{feed_name}-{file_hash}"

    def add_episode(
        self, feed_name: str, audio_file_path: str, metadata: PodcastMetadata
    ) -> Dict[str, str]:
        """
        Add a new episode to a feed.

        Args:
            feed_name: Name of the feed to add the episode to
            audio_file_path: Path to the audio file
            metadata: Episode metadata

        Returns:
            Dictionary containing the episode and feed URLs
        """
        # Ensure feed exists
        feed = self.db.get_feed(feed_name)
        if not feed:
            raise ValueError(f"Feed '{feed_name}' not found")

        # Extract audio metadata
        audio_metadata = self._get_audio_metadata(audio_file_path)

        # Generate file name and storage path
        file_name = os.path.basename(audio_file_path)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        storage_name = f"{timestamp}-{file_name}"
        storage_path = f"{self.audio_prefix}/{feed_name}/{storage_name}"

        # Upload audio file
        audio_url = self.storage.upload_file(
            file_path=audio_file_path,
            destination_path=storage_path,
            metadata={
                "feed_name": feed_name,
                "episode_title": metadata.title,
                "timestamp": timestamp,
            },
        )

        # Add episode to database
        episode = self.db.add_episode(
            feed_name=feed_name,
            title=metadata.title,
            description=metadata.description,
            audio_url=audio_url,
            guid=self._generate_guid(feed_name, audio_file_path),
            duration=metadata.duration or audio_metadata.get("duration"),
            episode_number=metadata.episode_number,
            season_number=metadata.season_number,
            episode_type=metadata.episode_type,
            explicit=metadata.explicit,
            image_url=metadata.image_url,
            publication_date=metadata.publication_date or datetime.now(),
            file_size=audio_metadata["file_size"],
            mime_type=audio_metadata.get("mime_type", "audio/mpeg"),
        )

        # Generate and upload new feed
        feed_xml = self.feed_generator.generate_feed(feed_name, self.base_url)
        feed_path = f"{self.feed_prefix}/{feed_name}.xml"
        feed_url = self.storage.upload_file(
            file_path=self._save_temp_feed(feed_xml),
            destination_path=feed_path,
            content_type="application/rss+xml",
        )

        return {"episode_url": audio_url, "feed_url": feed_url}

    def _save_temp_feed(self, feed_content: str) -> str:
        """Save feed XML to a temporary file."""
        temp_path = f"/tmp/feed-{datetime.now().timestamp()}.xml"
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(feed_content)
        return temp_path

    def create_feed(
        self, name: str, title: str, description: str, author: str, email: str, **kwargs
    ) -> str:
        """
        Create a new podcast feed.

        Args:
            name: Unique name for the feed
            title: Feed title
            description: Feed description
            author: Feed author name
            email: Author's email
            **kwargs: Additional feed parameters

        Returns:
            Feed URL
        """
        # Create feed in database
        feed = self.db.create_feed(
            name=name,
            title=title,
            description=description,
            author=author,
            email=email,
            **kwargs,
        )

        # Generate initial empty feed
        feed_xml = self.feed_generator.generate_feed(name, self.base_url)
        feed_path = f"{self.feed_prefix}/{name}.xml"

        # Upload feed
        feed_url = self.storage.upload_file(
            file_path=self._save_temp_feed(feed_xml),
            destination_path=feed_path,
            content_type="application/rss+xml",
        )

        return feed_url

    def get_feed_url(self, feed_name: str) -> str:
        """Get the URL for a feed."""
        return urljoin(self.base_url, f"{self.feed_prefix}/{feed_name}.xml")

    def list_episodes(
        self, feed_name: str, limit: Optional[int] = None
    ) -> list[Type[Episode]]:
        """List episodes in a feed."""
        return self.db.get_episodes(feed_name, limit)
