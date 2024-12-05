from email.utils import formatdate

import pytz
from feedgen.feed import FeedGenerator

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

        if feed_data.image_url is not None:
            fg.logo(feed_data.image_url)
            fg.image(feed_data.image_url)

        # iTunes specific tags
        fg.podcast.itunes_category(feed_data.categories.split(",")[0] if feed_data.categories else "Technology")  # pyright: ignore [reportAttributeAccessIssue, reportGeneralTypeIssues]
        fg.podcast.itunes_explicit(feed_data.explicit)  # pyright: ignore [reportAttributeAccessIssue, reportGeneralTypeIssues]
        fg.podcast.itunes_author(feed_data.author)  # pyright: ignore [reportAttributeAccessIssue, reportGeneralTypeIssues]
        fg.podcast.itunes_owner(name=feed_data.author, email=feed_data.email)  # pyright: ignore [reportAttributeAccessIssue, reportGeneralTypeIssues]

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
            fe.podcast.itunes_duration(str(episode.duration) if episode.duration is not None else "0")  # pyright: ignore [reportAttributeAccessIssue]
            fe.podcast.itunes_explicit(episode.explicit)  # pyright: ignore [reportAttributeAccessIssue]
            if episode.image_url is not None:
                fe.podcast.itunes_image(episode.image_url)  # pyright: ignore [reportAttributeAccessIssue]
            if episode.episode_number is not None:
                fe.podcast.itunes_episode(str(episode.episode_number))  # pyright: ignore [reportAttributeAccessIssue]
            if episode.season_number is not None:
                fe.podcast.itunes_season(str(episode.season_number))  # pyright: ignore [reportAttributeAccessIssue]
            fe.podcast.itunes_episode_type(episode.episode_type)  # pyright: ignore [reportAttributeAccessIssue]

        return fg.rss_str(pretty=True).decode("utf-8")
