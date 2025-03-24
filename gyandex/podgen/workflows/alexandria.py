from json import JSONDecodeError
from textwrap import dedent
from typing import List

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.exceptions import OutputParserException
from rich import print as rprint
from tenacity import retry, retry_if_exception_type, stop_after_attempt

from ...llms.factory import get_model
from ...loaders.types import Document
from ..config.schema import LLMConfig, Participant, PodcastConfig
from .types import ContentAnalysis, OutlineSegment, PodcastEpisode, PodcastOutline, ScriptSegment


class OutlineGenerator:
    def __init__(self, config: LLMConfig):
        self.model = get_model(config)

        self.parser = PydanticOutputParser(pydantic_object=PodcastOutline)

        # Create parser for content analysis
        self.analysis_parser = PydanticOutputParser(pydantic_object=ContentAnalysis)

        # Content analysis prompt to help LLM assess complexity and scale response appropriately
        self.content_analysis_prompt = PromptTemplate(
            template=dedent("""
            <role>
            You are an expert content analyst specializing in podcast production and content optimization.
            </role>

            <task>
            Analyze the provided content to determine the optimal structure for a podcast episode.
            </task>

            <content>
            Title: {title}

            Content:
            {content}
            </content>

            <reasoning>
            Think step-by-step:
            1. Assess the conceptual complexity of the material (technical terms, abstract concepts)
            2. Evaluate information density (amount of distinct ideas per paragraph)
            3. Identify and count the major distinct concepts presented
            4. Determine how many separate topics or themes the content covers
            5. Based on all factors, calculate the optimal number of podcast segments (3-8)
            </reasoning>

            <requirements>
            Provide a structured assessment with these metrics:
            1. Content Complexity: Rate from 1-5 (1=very simple, 5=highly complex)
            2. Information Density: Rate from 1-5 (1=sparse, 5=extremely dense)
            3. Concept Count: Estimate how many distinct key concepts are presented
            4. Content Breadth: How many separate topics or themes does this cover?
            5. Recommended Segments: Based on your analysis, how many podcast segments would be ideal (3-8)
            </requirements>

            <output_format>
            {format_instructions}
            </output_format>
            """),
            input_variables=["content", "title"],
            partial_variables={"format_instructions": self.analysis_parser.get_format_instructions()},
        )

        # Updated outline prompt with content-aware guidance
        self.outline_prompt = PromptTemplate(
            template=dedent("""
            <role>
            You are an expert podcast producer skilled at transforming written content into engaging, structured
            podcast outlines.
            </role>

            <context>
            Content Analysis Results:
            {content_analysis}

            Based on this analysis, aim for approximately {optimal_segments} segments.
            </context>

            <content>
            Title: {title}

            Content:
            {content}
            </content>

            <approach>
            Think step-by-step:
            1. Identify the core message and key themes in the content
            2. Group related ideas into potential segments
            3. Prioritize topics based on importance, complexity, and audience value
            4. Structure segments from foundational to advanced concepts
            5. Create natural transitions between segments
            6. Balance segment durations according to topic complexity
            </approach>

            <principles>
            1. Each segment must focus on a UNIQUE aspect with NO overlap
            2. Design progressive complexity flow (basic -> advanced -> application)
            3. Balance content types (conceptual, technical, practical, analytical, narrative)
            4. Create engaging hooks for each segment
            5. Maintain natural conversation flow between segments
            </principles>

            <requirements>
            Your outline must include:
            1. An overall podcast title that captures the essence of the content
            2. A compelling podcast description (2-3 sentences)
            3. Approximately {optimal_segments} segments with:
               - Clear segment name
               - 3-5 unique talking points per segment
               - Suggested duration (in minutes)
               - Natural transition to the next segment
            4. Total podcast duration proportional to content complexity
            </requirements>

            <self_critique>
            Before finalizing, verify:
            1. Is each segment focused on a unique aspect with no overlapping talking points?
            2. Does the flow progress naturally from foundational to complex topics?
            3. Are all major concepts from the source content included?
            4. Are segment durations appropriate for the topic complexity?
            5. Do transitions feel natural between segments?
            </self_critique>

            <output_format>
            {format_instructions}
            </output_format>
            """),
            input_variables=["content", "title", "content_analysis", "optimal_segments"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
        )

    def generate_outline(self, document: Document) -> PodcastOutline:
        """Generate structured podcast outline from content summary using two-stage prompting"""
        # Stage 1: Analyze content complexity and structure
        analysis_chain = (
            self.content_analysis_prompt
            | self.model
            | self.analysis_parser.with_retry(
                stop_after_attempt=3,
                retry_if_exception_type=(JSONDecodeError, OutputParserException),
            )
        )

        # Get structured analysis result
        analysis_result = analysis_chain.invoke({"content": document.content, "title": document.title})
        optimal_segments = analysis_result.optimal_segments
        analysis_str = analysis_result.model_dump_json(indent=2)

        # Stage 2: Generate the outline using the content analysis
        chain = (
            self.outline_prompt
            | self.model
            | self.parser.with_retry(
                stop_after_attempt=3,
                retry_if_exception_type=(JSONDecodeError, OutputParserException),
            )
        )

        response = chain.invoke(
            {
                "content": document.content,
                "title": document.title,
                "content_analysis": analysis_str,
                "optimal_segments": optimal_segments,
            }
        )

        return response


class ScriptGenerator:
    def __init__(self, config: LLMConfig, participants: List[Participant], additional_instructions: str = ""):
        self.model = get_model(config)
        self.additional_instructions = additional_instructions

        self.parser = PydanticOutputParser(pydantic_object=ScriptSegment)

        self.segment_prompt = PromptTemplate(
            input_variables=[
                "segment_name",
                "talking_points",
                "duration",
                "source_content",
                "previous_content",
                "segment_number",
                "total_segments",
                "transition",
            ],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions(),
                "host_profiles": "\n".join([self.create_host_profile(participant) for participant in participants]),
                "additional_instructions": additional_instructions,
            },
            template=dedent("""
            # PODCAST DIALOGUE GENERATION

            <role>
            You are an expert podcast scriptwriter who specializes in creating authentic, engaging conversations that
            sound completely natural.
            </role>

            <context>
            Creating segment {segment_number}/{total_segments} on topic: "{segment_name}"
            Duration: {duration} minutes
            </context>

            <character_profiles>
            {host_profiles}

            {additional_instructions}
            </character_profiles>

            <requirements>
            Cover these talking points naturally:
            {talking_points}

            Connect to previous content:
            {previous_content}

            Draw accurately from source material:
            {source_content}

            Use this transition to next topic: {transition}
            </requirements>

            <constraints>
            - BANNED WORDS/PHRASES: "exactly", "that's right", "I agree" + expansion, validation phrases
            - REQUIRED ELEMENTS: 3+ disagreements, 4+ interruptions, speech disfluencies, uneven turns
            - SEGMENT POSITION:
              * If segment 1: Begin with casual introduction
              * If middle segment: Create seamless flow from previous segments (no "welcome back" or "let's continue"
              or "wrapping up")
              * If final segment: End naturally with wrap-up and thanks
            </constraints>

            <reasoning>
            Let's think step-by-step:

            1. Analyze host personalities and their natural speaking patterns
               * How would each host uniquely approach this topic?
               * Where would their perspectives naturally align or differ?

            2. Explore multiple conversation paths (choose most authentic):
               * Path A: Start with agreement, then discover disagreement
               * Path B: Begin with contrasting views, find common ground
               * Path C: Explore tangent, then refocus on key points

            3. Plan natural interruptions and speech patterns:
               * Identify 3-4 moments for authentic interruptions
               * Determine places for speech disfluencies (um, uh, like)
               * Create 2-3 instances where hosts talk over each other briefly

            4. Structure conversation flow:
               * Create extended disagreements (5+ exchanges) with multiple layers
               * Use diverse evidence types (personal experience, expert opinions, logical reasoning)
               * Balance technical depth with accessible explanations
               * Include tangents that don't perfectly connect to main topics
            </reasoning>

            <self_critique>
            After drafting, verify:
            1. Does each line sound like something the host would naturally say?
            2. Are there enough natural speech patterns (hesitations, interruptions)?
            3. Have I avoided all banned phrases and artificial dialogue?
            4. Does the conversation cover all required points while feeling spontaneous?
            5. Are hosts' speaking styles distinct and consistent?
            6. Does the conversation flow naturally without obvious transitions?
            7. Have I created a distinct ending type for this segment?
            </self_critique>

            <output_format>
            Return ONLY valid JSON matching this structure:
            {format_instructions}
            </output_format>
            """),
        )

        self.chain = (
            self.segment_prompt
            | self.model
            | self.parser.with_retry(
                stop_after_attempt=3,
                retry_if_exception_type=(JSONDecodeError, OutputParserException),
            )
        )

    def create_host_profile(self, participant: Participant):
        return f"HOST ({participant.name})[{participant.gender}]: {participant.personality}"

    def _format_previous_segments(self, segments: List[ScriptSegment]) -> str:
        """Format previous segments into a readable context for the LLM"""
        if not segments:
            return ""

        context = ""

        for i, segment in enumerate(segments):
            context += f"\n--- SEGMENT {i + 1}: {segment.name} ---\n"
            for line in segment.dialogue:
                context += f"{line.speaker}: {line.text}\n"

        return context

    @retry(
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type((OutputParserException, JSONDecodeError)),
        reraise=True,
    )
    async def generate_segment_script(
        self,
        segment: OutlineSegment,
        source_content: str,
        segment_number: int,
        total_segments: int,
        previous_content="",
    ) -> ScriptSegment:
        """Generate script for a single segment using previous segments as context"""

        result = await self.chain.ainvoke(
            {
                "segment_name": segment.name,
                "talking_points": segment.talking_points,
                "duration": segment.duration,
                "source_content": source_content,
                "segment_number": segment_number,
                "total_segments": total_segments,
                "previous_content": previous_content,
                "transition": segment.transition,
            }
        )

        return result

    async def generate_full_script(self, outline: PodcastOutline, document_content: str) -> List[ScriptSegment]:
        """Generate full script with context awareness to avoid duplication"""
        results = []
        segments = outline.segments
        num_segments = len(segments)

        # Generate segments sequentially to maintain context
        for i, segment in enumerate(segments):
            # Format all previous segments as context
            previous_content = "NO PREVIOUS SEGMENTS, THIS IS THE FIRST SEGMENT"
            if results:  # If we have previous segments
                previous_content = self._format_previous_segments(results)

            script = await self.generate_segment_script(
                segment,
                document_content,
                segment_number=i + 1,
                total_segments=num_segments,
                previous_content=previous_content,
            )

            results.append(script)

        return results


class AlexandriaWorkflow:
    config: PodcastConfig

    def __init__(self, config: PodcastConfig):
        self.config = config

    async def generate_script(self, document: Document) -> PodcastEpisode:
        # Initialize components with enhanced configuration
        outline_gen = OutlineGenerator(self.config.workflow.outline)

        # Get additional_instructions if available in OpenAI TTS config
        additional_instructions = getattr(self.config.tts, "additional_instructions", "")
        script_gen = ScriptGenerator(self.config.workflow.script, self.config.tts.participants, additional_instructions)

        # Generate outline
        outline = outline_gen.generate_outline(document)

        # Pretty print the structured output
        if self.config.workflow.verbose:
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

        # Generate script segments with context awareness
        script_segments = await script_gen.generate_full_script(outline, document.content)

        if self.config.workflow.verbose:
            # Print results in dialogue format
            for i, segment in enumerate(script_segments):
                rprint(
                    f"\n[bold magenta]=== Segment {i + 1}: {segment.name} ({segment.duration} minutes) ===[/bold magenta]"  # noqa: E501
                )
                for line in segment.dialogue:
                    rprint(f"[bold]{line.speaker}:[/bold] {line.text}")
                rprint("")

        # Flatten script segments into dialogues
        segments = []
        for segment in script_segments:
            segments += segment.dialogue

        if self.config.workflow.verbose:
            rprint(f"[green]Successfully generated podcast with {len(segments)} dialogue lines![/green]")

        return PodcastEpisode(title=outline.title, description=outline.description, dialogues=segments)
