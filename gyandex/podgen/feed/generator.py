from feedgen.feed import FeedGenerator
from email.utils import formatdate
import pytz
from .models import PodcastDB


class PodcastFeedGenerator:
    def __init__(self, db: PodcastDB):
        self.db = db

    def generate_feed(self, slug: str) -> str:
        """
        Generate a podcast RSS feed.

        Args:
            slug: Slug of the feed to generate

        Returns:
            RSS feed XML as string
        """
        feed_data = self.db.get_feed(slug)
        if not feed_data:
            raise ValueError(f"Feed '{slug}' not found")

        fg = FeedGenerator()
        fg.load_extension("podcast")

        # Set feed level information
        fg.title(feed_data.title)
        fg.description(feed_data.description)
        fg.author({"name": feed_data.author, "email": feed_data.email})
        fg.link(href=feed_data.website, rel="alternate")
        fg.language(feed_data.language)
        fg.copyright(feed_data.copyright)

        if feed_data.image_url:
            fg.logo(feed_data.image_url)
            fg.image(feed_data.image_url)

        # iTunes specific tags
        fg.podcast.itunes_category(
            feed_data.categories.split(",")[0] if feed_data.categories else "Technology"
        )
        fg.podcast.itunes_explicit(feed_data.explicit)
        fg.podcast.itunes_author(feed_data.author)
        fg.podcast.itunes_owner(name=feed_data.author, email=feed_data.email)

        # Add episodes
        episodes = self.db.get_episodes(slug)
        for episode in episodes:
            fe = fg.add_entry()
            fe.id(episode.guid)
            fe.title(episode.title)
            fe.description(episode.description)

            # Format the publication date as RFC 2822
            pub_date = episode.publication_date.replace(tzinfo=pytz.UTC)
            fe.published(formatdate(float(pub_date.strftime("%s"))))

            # Add the audio enclosure
            fe.enclosure(episode.audio_url, str(episode.file_size), episode.mime_type)

            # iTunes specific episode tags
            fe.podcast.itunes_duration(
                str(episode.duration) if episode.duration else "0"
            )
            fe.podcast.itunes_explicit(episode.explicit)
            if episode.image_url:
                fe.podcast.itunes_image(episode.image_url)
            if episode.episode_number:
                fe.podcast.itunes_episode(str(episode.episode_number))
            if episode.season_number:
                fe.podcast.itunes_season(str(episode.season_number))
            fe.podcast.itunes_episode_type(episode.episode_type)

        return fg.rss_str(pretty=True).decode("utf-8")
