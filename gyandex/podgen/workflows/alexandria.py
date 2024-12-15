import asyncio
from json import JSONDecodeError
from textwrap import dedent
from typing import List

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.exceptions import OutputParserException
from rich import print as rprint

from ...llms.factory import get_model
from ...loaders.types import Document
from ..config.schema import LLMConfig, Participant, PodcastConfig
from .types import OutlineSegment, PodcastEpisode, PodcastOutline, ScriptSegment


class OutlineGenerator:
    def __init__(self, config: LLMConfig):
        self.model = get_model(config)

        self.parser = PydanticOutputParser(pydantic_object=PodcastOutline)

        self.outline_prompt = PromptTemplate(
            template=dedent("""
            Create a focused podcast outline based on the content

            Rules:
            1. Target podcast duration and number of segments should be proportional to the content length; 
               it should not be more than reading the content directly
            2. Each segment must focus on a UNIQUE aspect with NO overlap
            3. Keep segments concise and focused on actual content from the source
            4. Don't add speculative content or expand beyond the source material
            5. Talking points should be mutually exclusive across segments
            6. Maintain natural conversation flow between segments
            7. Explore different perspectives, so that important topics are covered holistically
            
            <title>{title}</title>
            <content>
            {content}
            </content>

            {format_instructions} 
            
            Make sure each segment has a clear transition to the next topic.
            """),
            input_variables=["content"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )

    def generate_outline(self, document: Document) -> PodcastOutline:
        """Generate structured podcast outline from content summary"""
        chain = (
            self.outline_prompt
            | self.model
            | self.parser.with_retry(
                stop_after_attempt=2,
                retry_if_exception_type=(JSONDecodeError, OutputParserException),
            )
        )
        response = chain.invoke({"content": document.content, "title": document.title})
        return response


class ScriptGenerator:
    def __init__(self, config: LLMConfig, participants: List[Participant]):
        self.model = get_model(config)

        self.parser = PydanticOutputParser(pydantic_object=ScriptSegment)

        self.segment_prompt = PromptTemplate(
            input_variables=["segment_name", "talking_points", "duration", "source_content"],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions(),
                "host_profiles": "\n".join([self.create_host_profile(participant) for participant in participants]),
            },
            template=dedent("""
            You are the a world-class podcast writer, you have worked as a ghost writer for Joe Rogan, 
            Lex Fridman, Ben Shapiro, Tim Ferris.
            We are in an alternate universe where actually you have been writing every line they say and 
            they just stream it into their brains.
            You have won multiple podcast awards for your writing.
            
            IMPORTANT: You are generating dialogue for the {position}

            Generate a podcast script segment as a dialogue between the following hosts:
            {host_profiles}

            SOURCE MATERIAL:
            <content>
            {source_content}
            </content>
            
            SEGMENT DETAILS:
            Topic: {segment_name}
            Key Points: {talking_points}
            Transition: {transition}

            DIALOGUE GENERATION RULES:
            1. Create natural dialogue with occasional fillers (um, uh, you know)
            2. Keep the dialogue flowing as one continuous conversation. 
               Keep it extremely engaging, the speakers can get derailed now and then but should discuss the topic. 
            3. If this is middle segment: let the conversation flow naturally into the next topic without 
               announcing transitions or welcoming statements
            4. End segment dialogues by building on the current point and naturally introducing elements of the 
               next topic
            5. If this is the closing segment, end the segment with a natural conclusion

            REQUIREMENTS:
            1. Generate text without special formatting, so that a TTS can vocalize it. 
               That means no asterisks or hyphens.
            2. Rewrite acronyms and abbreviations as full words, so that they are easier to pronounce.

            TRANSITION STYLE GUIDE:
            - Avoid phrases like "segues into" or "next topic"
            - Connect topics through shared themes or related ideas
            - Use natural conversational bridges like "That reminds me of..." or 
              "You know what's interesting about that..."
            - Let one host's insight naturally lead to the next area of discussion

            {format_instructions}
            """),
        )

        self.chain = (
            self.segment_prompt
            | self.model
            | self.parser.with_retry(
                stop_after_attempt=2,
                retry_if_exception_type=(JSONDecodeError, OutputParserException),
            )
        )

    def create_host_profile(self, participant: Participant):
        return f"HOST ({participant.name})[{participant.gender}]: {participant.personality}"

    async def generate_segment_script(
        self, segment: OutlineSegment, source_content: str, is_first=False, is_last=False, transition=""
    ) -> ScriptSegment:
        """Generate script for a single segment"""
        position = "opening segment" if is_first else "closing segment" if is_last else "middle segment"
        transition = transition if not is_last else ""
        result = await self.chain.ainvoke(
            {
                "segment_name": segment.name,
                "talking_points": segment.talking_points,
                "duration": segment.duration,
                "source_content": source_content,
                "position": position,
                "transition": transition,
            }
        )
        return result

    async def generate_full_script(self, outline: PodcastOutline, document_content: str) -> List[ScriptSegment]:
        """Generate all script segments with proper transitions"""
        segments = outline.segments
        tasks = []

        for i, segment in enumerate(segments):
            is_first = i == 0
            is_last = i == len(segments) - 1
            transition = segment.transition if not is_last else ""

            tasks.append(
                self.generate_segment_script(
                    segment,
                    document_content,
                    is_first=is_first,
                    is_last=is_last,
                    transition=transition,
                )
            )

        return await asyncio.gather(*tasks)


class AlexandriaWorkflow:
    config: PodcastConfig

    def __init__(self, config: PodcastConfig):
        self.config = config

    async def generate_script(self, document: Document) -> PodcastEpisode:
        # Initialize components
        outline_gen = OutlineGenerator(self.config.workflow.outline)
        script_gen = ScriptGenerator(self.config.workflow.script, self.config.tts.participants)

        # Generate outline
        outline = outline_gen.generate_outline(document)

        # Pretty print the structured output
        if self.config.workflow.verbose:
            rprint("[bold green]Generated Podcast Outline:[/bold green]")
            rprint(f"Title: {outline.title}")
            rprint(f"Description: {outline.description}")
            rprint(f"Duration: {outline.total_duration} minutes\n")
            rprint("[bold green]Generated Podcast Outline:[/bold green]")
            rprint(f"Title: {outline.title}")
            rprint(f"Description: {outline.description}")
            rprint(f"Duration: {outline.total_duration} minutes\n")

            for i, segment in enumerate(outline.segments, 1):
                rprint(f"[bold blue]Segment {i}: {segment.name}[/bold blue]")
                rprint(f"Duration: {segment.duration} minutes")
                rprint("Talking Points:")
                for point in segment.talking_points:
                    rprint(f"• {point}")
                rprint(f"Transition: {segment.transition}\n")

        # Generate script segments
        script_segments = await script_gen.generate_full_script(outline, document.content)

        if self.config.workflow.verbose:
            # Print results in dialogue format
            for segment in script_segments:
                print(f"\n=== {segment.name} ({segment.duration} minutes) ===")
                for line in segment.dialogue:
                    print(f"\n{line.speaker}: {line.text}")

        segments = []
        for segment in script_segments:
            segments += segment.dialogue
        rprint(f"Number of segments: {len(segments)}")
        return PodcastEpisode(title=outline.title, description=outline.description, dialogues=segments)
