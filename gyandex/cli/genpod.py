import argparse
import hashlib
import os
from collections import namedtuple

from dotenv import load_dotenv
from rich.console import Console

from gyandex.llms.factory import get_model
from gyandex.loaders.factory import load_content
from gyandex.podgen.engine.publisher import PodcastPublisher, PodcastMetadata
from gyandex.podgen.feed.models import PodcastDB
from gyandex.podgen.config.loader import load_config
from gyandex.podgen.engine.synthesizer import TTSEngine
from gyandex.podgen.engine.workflows import create_script
from gyandex.podgen.storage.factory import get_storage


def main():
    """Entry point for the CLI tool"""
    load_dotenv()

    parser = argparse.ArgumentParser(description="Generate a podcast")
    parser.add_argument("config_path", help="Path to the podcast config file")
    args = parser.parse_args()

    if args.config_path == "--help" or args.config_path == "--version":
        parser.print_help()
        return
    console = Console()
    config = load_config(args.config_path)
    model = get_model(config.llm)

    # Load the content
    with console.status('[bold green] Loading content...[/bold green]'):
        document = load_content(config.content)
    console.log('Content loaded...')

    # Analyze the content
    with console.status('[bold green] Crafting the script...[/bold green]'):
        script = create_script(model, document)  # attach callback to see the progress
    console.log(f'Script completed for "{script.title}". Script contains {len(script.segments)} segments...')

    # Generate the podcast audio
    with console.status('[bold green] Generating audio...[/bold green]'):
        tts_engine = TTSEngine()
        audio_segments = [tts_engine.process_segment(segment) for segment in script.segments]

        # Create output directory
        output_dir = f"generated_podcasts/{config.feed.slug}"
        os.makedirs(output_dir, exist_ok=True)

        podcast_path = f"{output_dir}/podcast_{hashlib.md5(config.content.source.encode()).hexdigest()}.mp3"
        tts_engine.generate_audio_file(audio_segments, podcast_path)
    console.log(f'Podcast file {podcast_path} generated...')

    with console.status('[bold green] Publishing podcast...[/bold green]'):
        storage = get_storage(config.storage)
        db = PodcastDB(db_path='assets/podcasts.db')
        publisher = PodcastPublisher(
            storage=storage,
            db=db,
            base_url=f"https://{storage.custom_domain}",  # @FIXME: we need to fallback when custom domain is not available
        )
        feed_url = publisher.create_feed(
            slug=config.feed.slug,
            title=config.feed.title,
            email=config.feed.email,
            website=str(config.feed.website),
            description=config.feed.description,
            author=config.feed.author,
            image_url=str(config.feed.image),
            language=config.feed.language,
            categories=",".join(config.feed.categories),
        )
        console.log('Uploading episode...')
        urls = publisher.add_episode(
            feed_slug=config.feed.slug,
            audio_file_path=podcast_path,
            metadata=PodcastMetadata(
                title=script.title,
                description=script.description,
            )
        )
    console.log(f"Feed published at {urls['feed_url']}")
    console.log(f"Episode published at {urls['episode_url']}")
