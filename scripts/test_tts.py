#!/usr/bin/env python3
"""
TTS Testing Script for Gyandex

This script loads a config YAML file and tests the TTS integration by generating
sample audio files using the configured TTS provider and voice settings.
It creates sample dialogue lines that demonstrate proper podcast opening,
natural conversation, and closing segments based on the Alexandria workflow.
"""

import argparse
import os
import random

from dotenv import load_dotenv
from rich.console import Console

from gyandex.podgen.config.loader import load_config
from gyandex.podgen.speech.factory import get_text_to_speech_engine
from gyandex.podgen.workflows.types import DialogueLine


def generate_sample_dialogues(participant_names):
    """
    Generate sample dialogue lines for testing TTS integration.

    Creates dialogue lines that demonstrate proper podcast opening,
    natural conversation, and closing segments following the Alexandria workflow.

    Args:
        participant_names: List of participant names from the config

    Returns:
        List of DialogueLine objects for testing TTS
    """
    if len(participant_names) < 2:
        # If only one participant, duplicate them for testing purposes
        participant_names = participant_names * 2

    # Ensure we have at least 2 participants for dialogue
    host1, host2 = participant_names[0], participant_names[1]

    dialogues = []

    # Opening segment - proper introduction following Alexandria workflow improvements
    dialogues.append(
        DialogueLine(
            speaker=host1,
            text="Welcome to our podcast! I'm your host, and today we're exploring a fascinating topic that intersects "
            "technology and everyday life.",
        )
    )

    dialogues.append(
        DialogueLine(
            speaker=host2,
            text="Great to be here! I'm excited to dive into this discussion with you and share some insights from my "
            "experience in this field.",
        )
    )

    # Middle segments - demonstrating natural dialogue patterns
    natural_responses = [
        "I think that's an important perspective, and it reminds me of something I read recently.",
        "You raise a good point. If we look at this from another angle, we might see additional factors at play.",
        "That's interesting. Let me build on that idea with an example from my own experience.",
        "I see what you're saying, though I wonder if we should also consider the broader implications.",
    ]

    questions = [
        "How do you think this affects people in their daily lives?",
        "Could you elaborate on how this technology might evolve in the next few years?",
        "What would you say is the most misunderstood aspect of this topic?",
        "Do you see any ethical considerations we should address here?",
    ]

    # Add some middle dialogue with natural conversation patterns
    dialogues.append(
        DialogueLine(
            speaker=host1,
            text="One thing that stands out to me is how rapidly this field has evolved. Just five years ago, nobody "
            "would have predicted where we are today.",
        )
    )

    dialogues.append(DialogueLine(speaker=host2, text=random.choice(natural_responses)))

    dialogues.append(
        DialogueLine(
            speaker=host2,
            text="When I think about the practical applications, I'm particularly interested in how this "
            "helps solve real-world problems.",
        )
    )

    dialogues.append(DialogueLine(speaker=host1, text=random.choice(questions)))

    dialogues.append(
        DialogueLine(
            speaker=host2,
            text="That's a thoughtful question. I believe the most significant impact will be in how it transforms "
            "our relationship with information and decision-making.",
        )
    )

    # Closing segment - proper conclusion following Alexandria workflow improvements
    dialogues.append(
        DialogueLine(
            speaker=host1,
            text="As we wrap up today's discussion, I'd like to highlight a few key takeaways. We've explored the "
            "evolution of this technology, its practical applications, and potential future developments.",
        )
    )

    dialogues.append(
        DialogueLine(
            speaker=host2,
            text="I think listeners should remember that this is an evolving field, and staying curious "
            "and informed is the best approach. Thank you all for joining us today!",
        )
    )

    dialogues.append(
        DialogueLine(
            speaker=host1,
            text="Thanks for listening, everyone. If you enjoyed this episode, please subscribe and join us next "
            "time for another fascinating discussion.",
        )
    )

    return dialogues


def main():
    """Entry point for the TTS testing script"""
    load_dotenv()  # Load environment variables from .env file

    parser = argparse.ArgumentParser(description="Test Text-to-Speech integration for Gyandex")
    parser.add_argument("config_path", help="Path to the podcast config file")
    parser.add_argument("--output-dir", "-o", default="tts_test_output", help="Directory to save generated audio files")
    args = parser.parse_args()

    console = Console()

    # Load the configuration file
    try:
        with console.status("[bold green]Loading configuration...[/bold green]"):
            config = load_config(args.config_path)
        console.print("[bold green]✓[/bold green] Configuration loaded successfully")
    except Exception as e:
        console.print(f"[bold red]Error loading configuration:[/bold red] {str(e)}")
        return

    # Check if TTS is enabled in the config
    if not config.tts.enabled:
        console.print("[bold yellow]TTS is disabled in the configuration.[/bold yellow]")
        console.print("Enable TTS in your config file and try again.")
        return

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Initialize TTS engine
    with console.status("[bold green]Initializing TTS engine...[/bold green]"):
        try:
            tts_engine = get_text_to_speech_engine(config.tts)
            console.print(
                f"[bold green]✓[/bold green] TTS engine initialized: [bold]{config.tts.provider}[/bold]"
                f"([italic]{config.tts.model}[/italic])"
            )
        except Exception as e:
            console.print(f"[bold red]Error initializing TTS engine:[/bold red] {str(e)}")
            return

    # Generate sample dialogues
    participant_names = [p.name for p in config.tts.participants]
    with console.status("[bold green]Generating sample dialogues...[/bold green]"):
        dialogues = generate_sample_dialogues(participant_names)
        console.print(f"[bold green]✓[/bold green] Generated {len(dialogues)} sample dialogue lines")

    # Generate individual audio files
    console.print("\n[bold]Generating individual audio samples:[/bold]")

    with console.status("[bold green]Processing dialogue lines...[/bold green]") as status:
        audio_segments = []

        for i, dialogue in enumerate(dialogues):
            try:
                status.update(f"[bold green]Processing line {i+1}/{len(dialogues)}...[/bold green]")

                # Clean the text for display
                display_text = dialogue.text[:40] + "..." if len(dialogue.text) > 40 else dialogue.text

                # Process the segment
                audio_data = tts_engine.process_segment(dialogue)
                audio_segments.append(audio_data)
                console.print(f"  [green]{i+1:02d}.[/green] [bold]{dialogue.speaker}:[/bold] {display_text}")
            except Exception as e:
                console.print(f"  [bold red]Error processing line {i+1}:[/bold red] {str(e)}")

    # Generate combined audio file
    with console.status("[bold green]Generating combined podcast sample...[/bold green]"):
        try:
            combined_path = f"{output_dir}/combined_podcast_sample.mp3"
            tts_engine.generate_audio_file(audio_segments, combined_path)
            console.print(f"[bold green]✓[/bold green] Combined audio saved to: [bold]{combined_path}[/bold]")
        except Exception as e:
            console.print(f"[bold red]Error generating combined audio:[/bold red] {str(e)}")

    console.print("\n[bold green]TTS testing complete![/bold green]")
    console.print(f"Combined podcast saved to: [bold]{output_dir}/[/bold]")


if __name__ == "__main__":
    main()
