import argparse
import asyncio
import hashlib
import os

from dotenv import load_dotenv
from rich.console import Console

from gyandex.loaders.factory import load_content
from gyandex.podgen.config.loader import load_config
from gyandex.podgen.engine.publisher import PodcastMetadata, PodcastPublisher
from gyandex.podgen.feed.models import PodcastDB
from gyandex.podgen.speech.factory import get_text_to_speech_engine
from gyandex.podgen.storage.factory import get_storage
from gyandex.podgen.workflows.factory import get_workflow


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

    # Load the content
    with console.status("[bold green] Loading content...[/bold green]"):
        document = load_content(config.content)
    console.log("Content loaded...")

    # Analyze the content
    with console.status("[bold green] Crafting the script...[/bold green]"):
        workflow = get_workflow(config)
        script = asyncio.run(workflow.generate_script(document))
    console.log(f'Script completed for "{script.title}". Script contains {len(script.dialogues)} segments...')

    # Create output directory
    output_dir = f"generated_podcasts/{config.feed.slug}"
    os.makedirs(output_dir, exist_ok=True)

    # Save the transcript
    transcript_path = f"{output_dir}/transcript_{hashlib.md5(config.content.source.encode()).hexdigest()}.txt"
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(f"Title: {script.title}\n")
        f.write(f"Description: {script.description}\n\n")
        f.write("Transcript:\n")
        for dialogue in script.dialogues:
            f.write(f"{dialogue.speaker}: {dialogue.text}\n")
    console.log(f"Transcript saved to {transcript_path}...")

    # Check if TTS is enabled
    if not config.tts.enabled:
        console.log("[yellow]TTS is disabled, skipping audio generation and publishing...[/yellow]")
        return

    # Generate the podcast audio
    with console.status("[bold green] Generating audio...[/bold green]"):
        tts_engine = get_text_to_speech_engine(config.tts)
        audio_segments = [tts_engine.process_segment(dialogue) for dialogue in script.dialogues]

        podcast_path = f"{output_dir}/podcast_{hashlib.md5(config.content.source.encode()).hexdigest()}.mp3"
        tts_engine.generate_audio_file(audio_segments, podcast_path)
    console.log(f"Podcast file {podcast_path} generated...")

    # Check if storage is enabled
    if not config.storage.enabled:
        console.log("[yellow]Storage is disabled, skipping podcast publishing...[/yellow]")
        return

    # Publish the podcast
    with console.status("[bold green] Publishing podcast...[/bold green]"):
        storage = get_storage(config.storage)
        db = PodcastDB(db_path="assets/podcasts.db")
        publisher = PodcastPublisher(
            storage=storage,
            db=db,
            # @FIXME: we need to fallback when custom domain is not available
            base_url=f"https://{storage.custom_domain}",
        )
        publisher.create_feed(
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
        console.log("Uploading episode...")
        urls = publisher.add_episode(
            feed_slug=config.feed.slug,
            audio_file_path=podcast_path,
            metadata=PodcastMetadata(
                title=script.title,
                description=script.description + f"\n\nSource: {config.content.source}",
            ),
        )
    console.print(f"Feed published at {urls['feed_url']}")
    console.print(f"Episode published at {urls['episode_url']}")
