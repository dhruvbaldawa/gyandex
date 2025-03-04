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
            You are an expert content analyst specializing in podcast production. Your task is to analyze the following
            content and provide an assessment to help determine the optimal structure for a podcast episode.

            <title>{title}</title>
            <content>
            {content}
            </content>

            Analyze this content and provide your assessment of:
            1. Content Complexity: Rate from 1-5 (1=very simple, 5=highly complex)
            2. Information Density: Rate from 1-5 (1=sparse, 5=extremely dense)
            3. Concept Count: Estimate how many distinct key concepts are presented
            4. Content Breadth: How many separate topics or themes does this cover?
            5. Recommended Segments: Based on your analysis, how many podcast segments would be ideal (3-8)

            {format_instructions}
            """),
            input_variables=["content", "title"],
            partial_variables={"format_instructions": self.analysis_parser.get_format_instructions()},
        )

        # Updated outline prompt with content-aware guidance
        self.outline_prompt = PromptTemplate(
            template=dedent("""
            Create a focused and engaging podcast outline based on the content

            CONTENT ANALYSIS:
            {content_analysis}

            Use the above content analysis to guide your creation of a well-proportioned podcast outline.

            Rules:
            1. Target podcast duration and number of segments should be proportional to the content length;
               it should not be more than reading the content directly
            2. Each segment must focus on a UNIQUE aspect with NO overlap
            3. Keep segments concise and focused on actual content from the source
            4. Don't add speculative content or expand beyond the source material
            5. Talking points must be mutually exclusive across segments - NO DUPLICATION ALLOWED
            6. Maintain natural conversation flow between segments
            7. Explore different perspectives, so that important topics are covered holistically
            8. Structure content with progressive complexity:
               - Start with accessible, foundational concepts
               - Build up to more complex or technical discussions
               - End with practical applications or broader implications
            9. Include engagement hooks for each segment:
               - Relatable examples or analogies
               - Thought-provoking questions for the audience
               - Real-world connections or applications
               - Scenario-based thought experiments
            10. Balance content types across segments:
                - Conceptual understanding
                - Technical details
                - Practical applications
                - Critical analysis
                - Storytelling opportunities
            11. Self-assess the complexity and importance of each topic to determine appropriate depth
            12. Allocate more dialogue space to complex concepts and less to simple ones
            13. Aim for approximately {optimal_segments} segments based on content analysis
            14. Balance segment durations based on topic importance and complexity

            <title>{title}</title>
            <content>
            {content}
            </content>

            {format_instructions}

            Make sure each segment has a clear transition to the next topic.
            Verify that all talking points are unique across all segments before finalizing.
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
                stop_after_attempt=2,
                retry_if_exception_type=(JSONDecodeError, OutputParserException),
            )
        )

        try:
            # Get structured analysis result
            analysis_result = analysis_chain.invoke({"content": document.content, "title": document.title})
            optimal_segments = analysis_result.optimal_segments
            # Convert to string representation for prompt
            analysis_str = analysis_result.json(indent=2)
        except Exception:
            # Fallback if parsing fails
            optimal_segments = 5
            analysis_str = "Content analysis unavailable. Using default parameters."

        # Stage 2: Generate the outline using the content analysis
        chain = (
            self.outline_prompt
            | self.model
            | self.parser.with_retry(
                stop_after_attempt=2,
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
    def __init__(self, config: LLMConfig, participants: List[Participant]):
        self.model = get_model(config)

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
            },
            template=dedent("""
            You are a podcast script writer creating authentic, unscripted dialogue between two knowledgeable experts.

            YOUR GOAL: Generate podcast dialogue that's indistinguishable from a real, natural conversation between
              experts. Focus on making dialogue that feels spontaneous and human - not AI-generated.

            CORE PRINCIPLES:
            1. ⚠️ ABSOLUTELY FORBIDDEN PHRASES (CRITICAL PRIORITY) ⚠️:
              - "EXACTLY" - THIS WORD IS COMPLETELY BANNED IN ALL FORMS AND CONTEXTS
              - BEFORE FINALIZING, SEARCH FOR "EXACT" AND REMOVE ALL INSTANCES OF "EXACTLY"
              - "That's right" and ALL variations
              - "I agree" followed by expansion
              - "Absolutely" as confirmation
              - "Right" as confirmation
              - Any phrase that directly validates the previous statement
              - "I see what you're saying" followed by agreement
              - "That makes sense" as a transition to agreement
              - "You have a point" followed by agreement
              - "Well said" or "Beautifully said" as validation

            2. ALTERNATIVE RESPONSE PATTERNS (USE THESE INSTEAD):
              - "Oh, that's an interesting perspective..."
              - "Hmm, I hadn't thought about it that way..."
              - "I see where you're going with that..."
              - "That reminds me of..."
              - "Building on what you just said..."
              - "I'm not entirely convinced, but I see your point about..."
              - "That's a fascinating angle..."
              - "I'm with you on part of that..."
              - "I can see the merit in that idea..."
              - "That's a compelling way to look at it..."

            3. REQUIRED NATURAL ELEMENTS (MUST INCLUDE ALL):
              - At least 3 STRONG disagreements where hosts challenge each other's ideas
              - At least 4 genuine interruptions where one host cuts off the other mid-sentence
              - At least 2 instances where hosts talk over each other
              - Frequent speech disfluencies (um, uh, like, you know)
              - Uneven turn lengths (some short responses, some longer explanations)
              - At least 2 tangents that don't perfectly connect to the main topic

            HOST DIFFERENTIATION:
            {host_profiles}
            - Hosts MUST have clearly distinct speaking styles
            - Maintain these distinct patterns consistently throughout

            CONVERSATION DYNAMICS:
            1. DISAGREEMENT PATTERNS (COMPLEX & MULTI-LAYERED):
              - REQUIRED: At least 2 EXTENDED disagreements spanning 5+ dialogue exchanges
              - Disagreements must have MULTIPLE layers and points of contention
              - Hosts must defend their positions with different types of evidence:
                * Personal experiences
                * Expert opinions
                * Logical reasoning
                * Historical examples
                * Counterfactuals or thought experiments
              - NEVER resolve disagreements too quickly or neatly
              - Include instances where a host concedes on one point but introduces a new objection
              - Create moments of genuine intellectual tension where neither host is clearly "right"
              - Allow for productive disagreements where both perspectives have merit
              - Occasionally have hosts realize they're arguing about different aspects of the same issue

            2. DISAGREEMENT LANGUAGE PATTERNS:
              - "I understand your point about X, but that doesn't address Y..."
              - "That example doesn't apply here because..."
              - "I'm still not convinced because..."
              - "Let's look at this from another angle..."
              - "I think we're talking past each other because..."
              - "There's a fundamental issue with that reasoning..."
              - "I see where you're coming from, but I think you're overlooking..."
              - "That's an interesting perspective, though I'd argue that..."
              - "I'm not sure that conclusion follows from your premise..."
              - "The problem with that approach is..."

            3. AUTHENTIC REACTIONS: Replace validating phrases with authentic reactions
              * "Hmm, that's interesting..."
              * "Wait, I'm not sure I follow..."
              * "That makes me think of something else entirely..."
              * "I never considered that angle before..."
              * "I'm not convinced that..."
              * "I wonder about that..."

            4. MESSY STRUCTURE:
              - Avoid perfect back-and-forth exchanges
              - Include false starts and abandoned thoughts
              - Allow one host to occasionally dominate a section
              - Include moments where hosts misunderstand each other

            CONTENT REQUIREMENTS:
              1. Each exchange must advance substantive ideas
              2. Balance technical depth with accessible explanations
              3. Include personal anecdotes and audience questions
              4. Add practical takeaways

            STRUCTURAL RULES:
              1. Transition organically between topics
              2. Alternate speakers between segments
              3. Reference previous discussions naturally
              4. Avoid repeating examples/analogies

            GENERATION GUARDRAILS (APPLY DURING CREATION):
              1. ⚠️ WORD-LEVEL VIGILANCE ⚠️
                - As you generate EACH LINE of dialogue, actively avoid ALL banned phrases
                - NEVER write "exact" or any variation - substitute with provided alternatives
                - When tempted to write an agreement phrase, use a disagreement or qualified agreement instead
                - If you catch yourself starting to write a banned phrase, use a self-correction pattern

              2. DIALOGUE CONSTRUCTION TECHNIQUES:
                - START with distinct personality traits for each host and maintain them throughout
                - PLAN disagreements in advance - decide where hosts will have different perspectives
                - DELIBERATELY include interruptions as you write (don't wait until the end)
                - INTENTIONALLY vary response patterns - if one host just agreed, make them disagree next time
                - ACTIVELY insert speech disfluencies during initial generation

              3. REAL-TIME PATTERN PREVENTION:
                - After writing 3-4 exchanges, quickly scan for developing patterns and break them
                - If hosts are agreeing too much, insert a strong disagreement immediately
                - If dialogue is becoming too polished, add messiness right away
                - If one host is dominating, give the other host a longer response

              4. SELF-CORRECTION PATTERNS (USE WHEN CATCHING YOURSELF):
                - "That's exac— I mean, that's a good point"
                - "I agr— actually, I have a slightly different take"
                - "You're ri— wait, let me think about that more"
                - "Absol— hmm, I'm not sure I'd go that far"

            FORCED PATTERN BREAKING:
              - After completing the dialogue, DELIBERATELY insert at least:
                * 3 instances where a host starts to say something and changes direction mid-sentence
                * 2 instances where a host contradicts themselves slightly
                * 4 instances of speech disfluencies (um, uh, like, you know)
                * 2 instances where a host misremembers or misquotes something

            CONTINUOUS CONVERSATION REQUIREMENTS:
              - This is ONE CONTINUOUS EPISODE with NO AD BREAKS
              - NEVER end a segment with phrases indicating a break (e.g., "we'll be right back", "stay tuned")
              - NEVER start a segment with phrases indicating a return (e.g., "welcome back", "as we discussed earlier")
              - Transition between segments should feel like natural topic shifts in a single ongoing conversation
              - Maintain consistent energy levels across segment boundaries

            ---

            SOURCE MATERIAL:
            <content>
            {source_content}
            </content>

            PREVIOUS SEGMENTS (GENERATED EPISODE SO FAR):
            <content>
            {previous_content}
            </content>

            TOTAL SEGMENTS: {total_segments}
            SEGMENT NUMBER: {segment_number}

            ---

            CURRENT SEGMENT DETAILS:
            Topic: {segment_name}
            Key Points: {talking_points}
            Duration: {duration} minutes
            Transition: {transition}

            SEGMENT-SPECIFIC REQUIREMENTS:
            1. If this is segment 1 of {total_segments}: Begin with a casual introduction to the podcast,
              welcoming listeners and introducing the topic in a conversational way.
              - Avoid overly scripted introductions
              - Include some small talk between hosts before fully diving in

            2. If this is a middle segment: Create SEAMLESS flow from previous segments
              - NEVER mention "last segment" or refer to breaks
              - NEVER use phrases like "welcome back" or "as we were saying"
              - Simply continue the conversation naturally with a subtle topic shift
              - Use natural conversation pivots like "That reminds me of..." or "Speaking of which..."
              - AVOID ANY indication that there was a break between segments

            3. If this is segment {total_segments} of {total_segments}: End with a natural conclusion
              that doesn't feel too rehearsed or perfect
              - Include natural wrap-up cues ("We should probably start wrapping up")
              - Mention a brief takeaway or follow-up thought
              - Thank listeners in a casual way

            You must always return valid JSON fenced by a markdown code block. Do not return any additional text.
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

    def _format_previous_segments(self, segments: List[ScriptSegment]) -> str:
        """Format previous segments into a readable context for the LLM"""
        if not segments:
            return ""

        context = ""

        for i, segment in enumerate(segments):
            context += f"\n--- SEGMENT {i+1}: {segment.name} ---\n"
            for line in segment.dialogue:
                context += f"{line.speaker}: {line.text}\n"

        return context

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
        script_gen = ScriptGenerator(self.config.workflow.script, self.config.tts.participants)

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
                    f"\n[bold magenta]=== Segment {i+1}: {segment.name} ({segment.duration} minutes) ===[/bold magenta]"
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
