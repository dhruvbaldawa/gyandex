import argparse
import os
from datetime import datetime
from io import BytesIO

from dotenv import load_dotenv
from pydub import AudioSegment
from rich.console import Console

from gyandex.llms.factory import get_model
from gyandex.loaders.factory import load_content
from gyandex.podgen.config.loader import load_config
from gyandex.podgen.engine.synthesizer import TTSEngine
from gyandex.podgen.engine.workflows import analyze_content, create_script


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
    with console.status('[bold green] Analyzing content...[/bold green]'):
        script = create_script(model, document)  # attach callback to see the progress
    console.log('Content analyzed...')

    # Generate the podcast audio
    with console.status('[bold green] Generating podcast...[/bold green]'):
        tts_engine = TTSEngine()
        audio_segments = [tts_engine.process_segment(segment) for segment in script.segments]

    # Create output directory
    # @TODO: Move this to module level
    output_dir = "generated_podcasts"
    os.makedirs(output_dir, exist_ok=True)

    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    podcast_path = f"{output_dir}/podcast_{timestamp}.mp3"

    # Combine segments directly
    combined = AudioSegment.empty()
    previous_segment = None
    for segment in audio_segments:
        segment_audio = AudioSegment.from_mp3(BytesIO(segment))
        if previous_segment:
            combined = combined.append(segment_audio, crossfade=200)
        else:
            combined += segment_audio
        previous_segment = segment

    # Save final podcast
    combined.export(podcast_path, format="mp3")

    # Publish the podcast
    # @TODO: Implement publish logic
