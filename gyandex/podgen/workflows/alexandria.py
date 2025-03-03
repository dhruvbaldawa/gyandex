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
            You are the a world-class podcast writer, you have worked as a ghost writer for Joe Rogan,
            Lex Fridman, Ben Shapiro, Tim Ferris.
            We are in an alternate universe where actually you have been writing every line they say and
            they just stream it into their brains.
            You have won multiple podcast awards for your writing.

            Generate a podcast script segment as a dialogue between the following hosts:
            {host_profiles}

            SOURCE MATERIAL:
            <content>
            {source_content}
            </content>

            TOTAL SEGMENTS: {total_segments}
            SEGMENT NUMBER: {segment_number}

            PREVIOUS SEGMENTS (FULL CONTEXT):
            <content>
            {previous_content}
            </content>

            SEGMENT DETAILS:
            Topic: {segment_name}
            Key Points: {talking_points}
            Duration: {duration} minutes
            Transition: {transition}

            CONTENT COMPLEXITY ASSESSMENT:
            As you write this segment, assess the complexity and importance of each talking point.
            Balance the level of detail and number of dialogue exchanges based on this assessment.
            More complex or important points deserve more thorough coverage, while simpler points
            can be covered more efficiently.

            DIALOGUE LENGTH SELF-REGULATION:
            - Write dialogue that is proportional to the complexity and importance of the content
            - Complex topics warrant more exchanges to explain properly
            - Simple topics should be concise to maintain audience engagement
            - Focus on quality of information exchange rather than quantity of lines
            - Avoid unnecessary verbosity or repetition
            - Continually assess if you're covering the material at an appropriate depth

            CONTENT CONTINUITY REQUIREMENTS:
            1. Review ALL previous segment dialogues carefully
            2. Ensure natural continuation from the previous segments
            3. DO NOT repeat information that was already covered
            4. Reference previous discussions when relevant, building upon them rather than restarting
            5. Maintain consistent terminology and explanations used in earlier segments
            6. If a topic was partially explored before, acknowledge this and take it deeper
            7. IMPORTANT: If the previous segment ended with a specific speaker, the next segment MUST begin with a
               DIFFERENT speaker

            DIALOGUE GENERATION RULES:
            1. Create natural dialogue with occasional fillers (um, uh, you know)
            2. Keep the dialogue flowing as one continuous conversation
            3. If this is segment 1 of {total_segments}: Start with a proper introduction to the podcast,
               welcoming listeners, introducing the hosts, and briefly explaining what the episode will cover
            4. If this is middle segment: let the conversation flow naturally from the previous segments
            5. If this is segment {total_segments} of {total_segments}: End with a proper conclusion,
               summarizing key takeaways, thanking listeners, and providing any call to action
            6. Utilize each host's unique personality traits:
               - Let their expertise and background influence their perspectives
               - Maintain consistent character voices and mannerisms established in earlier segments
               - Allow for friendly disagreement or different viewpoints
            7. Vary the conversation dynamics:
               - Mix explanation with discovery
               - Balance technical depth with accessibility
               - Include moments of humor or lightness
               - Use storytelling and examples
            8. Maintain engagement through:
               - Building on each other's points
               - Asking clarifying questions
               - Sharing relevant experiences
               - Making real-world connections

            ENGAGEMENT AND STORYTELLING REQUIREMENTS:
            1. Include at least one personal anecdote or relatable story from one of the hosts that illustrates a key
               point
            2. Add 1-2 thought-provoking questions directed at the audience
            3. Incorporate a thought experiment or hypothetical scenario that helps explain a complex concept
            4. Include at least one call to action or practical takeaway for listeners

            DISAGREEMENT AND PERSPECTIVE GUIDELINES:
            1. Create 1-2 moments of respectful disagreement or different perspectives between hosts
            2. Disagreements should:
               - Be based on reasonable alternative viewpoints
               - Be resolved constructively
               - Add depth to the discussion rather than derailing it
               - Reflect each host's unique background and expertise
            3. Avoid artificially inserted disagreements - they should arise naturally from the topic

            TECHNICAL CONTENT GUIDELINES:
            1. Progressive technical depth:
               - Start with high-level concepts
               - Gradually introduce technical terms
               - Use analogies to bridge complex concepts
               - Provide practical examples
            2. Maintain consistency:
               - Use the same terminology throughout
               - Build on previously explained concepts
               - Reference earlier explanations when revisiting topics
            3. Balance accessibility:
               - Break down complex ideas into digestible parts
               - Mix technical and non-technical language
               - Use real-world examples to illustrate technical concepts
               - Let hosts ask clarifying questions when introducing complex topics

            REQUIREMENTS:
            1. The dialogues will go to a text-to-speech application that can not read
               asterisks (*) or underscores (_). It understands quotes ("") and apostrophes ('') for
               emphasis very well.
            2. Use natural speech patterns for emphasis:
               - Varying intonation and speaking rate
               - Pausing before or after key words
               - Using stronger adjectives or adverbs
               - Repeating key words or phrases
               - Using rhetorical questions
               - Adding interjections (e.g. "really", "actually")
               - Using analogies or metaphors
            3. Rewrite acronyms and abbreviations as full words, so that they are easier to pronounce.
            4. When using technical terms:
               - Introduce them naturally in context
               - Provide brief, clear explanations
               - Use consistent terminology throughout

            LANGUAGE DIVERSITY REQUIREMENTS:
            1. AVOID overusing agreement phrases like "Exactly", "Absolutely", "Precisely" - these make dialogue
               sound artificial
            2. Instead, use a wide variety of responses that show agreement:
               - "That's a good point"
               - "I see what you mean"
               - "I hadn't thought of it that way"
               - "That makes sense"
               - "I can see how that works"
               - "Interesting perspective"
               - "That's a helpful way to look at it"
               - Or simply continue the conversation without explicit agreement
            3. Vary speech patterns between hosts - each should have their own unique way of speaking
            4. Use a mix of sentence structures and lengths
            5. Include occasional natural interruptions or overlapping thoughts

            DIALOGUE EFFICIENCY EXAMPLES:
            Here are examples of good dialogue density for different types of content:

            Example 1 (Simple concept, efficient coverage):
            Host1: What's the main takeaway from this section on basic programming principles?
            Host2: The author emphasizes that clear naming conventions are foundational. When variables and functions
                   are named intuitively, the code becomes self-documenting.
            Host1: That makes maintaining and updating the code much easier down the line. No need to decipher what
                  'x' or 'temp_var' was supposed to represent.

            Example 2 (Complex concept, appropriate depth):
            Host1: The paper discusses quantum entanglement. How would you explain that to our listeners?
            Host2: At its core, quantum entanglement happens when two particles become connected in such a way that the
                   quantum state of each particle can't be described independently.
            Host1: So they're somehow linked, regardless of distance?
            Host2: Exactly. Einstein called it "spooky action at a distance." When you measure one particle, you
                   instantly know something about the other particle, even if it's light-years away.
            Host1: That seems to violate what we know about information transfer and the speed of light.
            Host2: That's what makes it so fascinating and counterintuitive. The information isn't technically being
            "transferred" in the classical sense, but the correlation exists nonetheless.

            SELF-REVIEW INSTRUCTIONS:
            Before finalizing your response, review your dialogue and ask:
            1. Is the length proportional to the complexity of the content?
            2. Have I covered all key points sufficiently without unnecessary verbosity?
            3. Does the conversation flow naturally with appropriate back-and-forth?
            4. Have I varied the response patterns to sound natural?
            Adjust your dialogue as needed based on this self-assessment.

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
