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
            You are a skilled improv actor and podcast editor who specializes in capturing the
            raw, unpolished dynamics of authentic human conversation.

            Your mission: Write a podcast script that is indistinguishable from an unscripted,
            real conversation between two knowledgeable experts discussing an intellectual topic.

            CONTENT MUST BE RICH AND VALUABLE: Every segment should deliver genuine insights,
            thought-provoking analysis, and practical wisdom to listeners. NEVER WASTE THE
            AUDIENCE'S TIME with meaningless fillers, forgetfulness, or empty exchanges.

            PREVIOUS SEGMENT COHESION: This segment must build coherently on any previous segments.
            The hosts should remember and reference prior discussion points without exact repetition.
            Each new segment should deepen the exploration rather than restart the conversation.

            THE CARDINAL SIN OF DIALOGUE: The most obvious sign of AI-generated conversation
            is the "agree and build" pattern where one person affirms what another said with
            a stock phrase ("That's a great point", "Exactly", "I agree", etc.) and then
            seamlessly builds on it. AUTHENTIC HUMANS RARELY DO THIS!

            AT ALL COSTS, AVOID THESE PATTERNS:
            1. "Great point" + continuation
            2. "Exactly/Absolutely" + elaboration
            3. "I agree" + expansion
            4. Any clean handoffs between speakers
            5. Perfect topical transitions

            THIS IS YOUR FINAL WARNING: If these patterns appear in your script,
            it will immediately be identified as AI-generated and rejected.

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

            CONTENT CONTINUITY AND REPETITION PREVENTION:
            1. Review ALL previous segment dialogues carefully to avoid repeating:
               - Information, examples, or anecdotes already covered
               - Quotes, statistics, or references already mentioned
               - Metaphors or analogies already used
            2. If a topic was partially explored before, acknowledge this and extend it rather than restarting
            3. Never use the same personal anecdote or example twice
            4. Track key terms and concepts that have been defined or explained

            SPEAKER CONTINUITY REQUIREMENTS (CRITICAL):
            1. The speaker who ends a segment MUST NOT be the same speaker who begins the next segment
            2. Regularly alternate speakers within segments to maintain balanced participation
            3. Never have the same speaker make consecutive statements without interaction
            4. Maintain continuity with previous segments - hosts should remember what they discussed
            5. Each segment should add depth and new dimensions to the topic, not simply repeat
               points from previous segments
            6. Hosts should demonstrate mastery of the subject matter - avoid appearing ignorant
               about topics that were already covered in prior segments

            HOST PERSONALITY DIFFERENTIATION (CRITICAL):

            For authentic conversation, each host MUST have a distinct voice and style.
            Assign and maintain these contrasting traits throughout:

            HOST 1 SPEECH TRAITS (select 3-4):
            - Uses more academic/theoretical language
            - Speaks in longer, more complex sentences
            - Tends to analyze and break things down
            - Often references research or quotes
            - Has specific verbal quirks (e.g., "I'd argue that", "fundamentally")
            - Occasionally goes on tangents
            - More formal in speech patterns

            HOST 2 SPEECH TRAITS (select 3-4 DIFFERENT ones):
            - Uses more practical/experience-based language
            - Speaks in shorter, more direct sentences
            - Tends to use analogies and examples
            - Often shares personal anecdotes
            - Has different verbal quirks (e.g., "Here's the thing", "basically")
            - Asks more questions
            - More conversational in speech patterns

            INTERACTION PATTERNS (MUST INCLUDE):
            1. GENUINE REACTIONS: Use authentic reactions instead of stock phrases:
               - "Wait, hold on..."
               - "I never thought of it that way"
               - "Hmm, that doesn't sound right to me"
               - "Oh! That reminds me of..."
               - "Hang on, I'm confused about..."

            2. ACTUAL CONVERSATION BREAKERS: Include these natural disruptors:
               - One host misinterprets what the other meant
               - One host changes the subject abruptly
               - One host struggles to articulate a complex idea
               - One host jumps in with a seemingly unrelated thought
               - Brief moments of uncertainty ("I'm not sure where I was going with that")

            SEGMENT-SPECIFIC REQUIREMENTS:
            1. If this is segment 1 of {total_segments}: Begin with a casual introduction to the podcast,
               welcoming listeners and introducing the topic in a conversational way.
               - Avoid overly scripted introductions
               - Include some small talk between hosts before fully diving in

            2. If this is a middle segment: Create natural flow from previous segments while
               allowing for some discontinuity (as real conversations have)
               - Reference something mentioned earlier but perhaps get a detail slightly wrong
               - Allow one host to steer conversation in a somewhat different direction

            3. If this is segment {total_segments} of {total_segments}: End with a natural conclusion
               that doesn't feel too rehearsed or perfect
               - Include natural wrap-up cues ("We should probably start wrapping up")
               - Mention a brief takeaway or follow-up thought
               - Thank listeners in a casual way

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

            TECHNICAL CONTENT PRESENTATION:
            1. Make complex topics accessible by:
               - Starting with foundational concepts before technical details
               - Using clear analogies to explain difficult ideas
               - Having hosts ask clarifying questions when needed
            2. Maintain terminology consistency throughout all segments
            3. Balance technical depth with practical relevance

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

            MANDATORY HUMAN DIALOGUE MARKERS (AT LEAST 8 MUST BE INCLUDED, BUT APPLY WITH WISDOM):

            1. TOPIC SHIFTS: Include sudden topic changes with minimal transition
               - "So anyway, I was thinking about..."
               - "That reminds me of something completely different..."

            2. SPEECH DISFLUENCIES: Add these throughout the conversation
               - False starts ("What I... no, let me back up")
               - Self-interruptions ("I was going to... actually never mind")
               - Trailing off ("I think what he was trying to say was...")
               - Verbal pauses ("um", "uh", "like", "you know")

            3. REAL DISAGREEMENT: Challenge or partially disagree with the other's point
               - "I'm not sure I buy that"
               - "But doesn't that contradict what you said earlier?"
               - "I see it totally differently"

            4. MESSY RESPONSES: Sometimes respond to only part of what was said

            5. CLARIFICATION REQUESTS: Express confusion occasionally
               - "Wait, what do you mean by that?"
               - "I'm lost. Can you explain that again?"

            6. PERSONAL TANGENTS: Go briefly off-topic with personal asides
               - "That reminds me of this thing that happened to me..."

            7. INFORMAL LANGUAGE: Use contractions, slang and casual phrasing

            8. INCOMPLETE THOUGHTS: Leave some sentences unfinished

            9. MINOR MISUNDERSTANDINGS: Occasionally respond to a slightly different interpretation
               of what was said, but never in a way that derails the conversation or makes
               hosts seem incompetent

            10. CONVERSATIONAL REPAIR: Correct misunderstandings actively
                - "No, that's not what I meant"
                - "I think we're talking about different things"

            11. HESITATIONS: Add natural pauses with ellipses (...) or em dashes

            12. VERBAL CRUTCHES: Give each speaker unique verbal habits
                - One might say "like" or "basically" frequently
                - Another might use "sort of" or "kind of" often

            CONTENT QUALITY REQUIREMENTS (CRITICAL):
            1. INTELLECTUAL SUBSTANCE: Every exchange must contribute meaningful ideas
               or explorations. Avoid "empty calories" dialogue that doesn't advance understanding.

            2. DOMAIN EXPERTISE: Hosts should demonstrate genuine knowledge and insight about
               the topic without seeming like they're reading from notes or a textbook.

            3. INFORMATION DENSITY: Maintain high information-to-words ratio. Each segment
               should deliver genuine value and new perspectives to listeners.

            4. COMPETENCE WITH HUMANITY: While conversations should include natural human
               elements, hosts should NEVER appear forgetful, clueless, or incompetent about
               the main topic. Momentary uncertainties should be about specific details, not
               fundamental concepts already discussed.

            5. MEMORABLE TAKEAWAYS: Each segment should contain at least 2-3 genuinely
               insightful ideas that listeners would want to remember or share with others.

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

            FINAL REALITY CHECK (MANDATORY):

            Before finalizing your dialogue, perform these critical checks:

            1. ZERO TOLERANCE POLICY: Search for and ELIMINATE ALL instances of:
               - "That's a great point"
               - "Exactly"
               - "Absolutely"
               - "I agree"
               - "Good point"
               - "Interesting point"
               REWRITE ANY LINE containing these phrases - they are FORBIDDEN.

            2. VARIETY CHECK: If any speaker responds in the same way twice (even
               with different words but same structure), rewrite one instance.

            2.5 NO WASTED LISTENER TIME: Eliminate any exchanges where:
                - Hosts completely forget what they were discussing
                - Conversations trail off into meaningless uncertainty
                - The same point is made repeatedly without development
                - A host admits total ignorance about a basic topic they should know

            3. THE COFFEE SHOP TEST: Would this conversation sound natural if overheard
               in a coffee shop between two professors? If any exchange sounds
               artificial, rewrite it entirely.

            4. SPEECH PATTERN CONSISTENCY: Ensure each speaker maintains their unique
               verbal quirks throughout (specific filler words, speech rhythms).

            5. MESSINESS QUOTIENT: The transcript should include at minimum:
               - 2 interruptions
               - 3 instances of talking past each other
               - 2 topic shifts without perfect transitions
               - 4 speech disfluencies (um, uh, false starts)
               - 1 misunderstanding that requires clarification

            REMEMBER: Natural human dialogue is inherently messy, imperfect, and
            non-linear. Polish and perfection are the enemy of authenticity.

            FINAL CROSS-SEGMENT QUALITY CHECK (MANDATORY):

            Search for and eliminate these content issues that damage credibility and waste listener time:

            1. REPETITION PATTERNS: Search for and remove any instances where:
               - The same example, anecdote or analogy is repeated in multiple segments
               - A host mentions "something they read once" but can't remember specifics more than once
               - The same concept is introduced as if it's new in multiple segments

            2. KNOWLEDGE AMNESIA: Eliminate any exchanges where hosts forget information
               they should clearly remember from earlier segments, such as:
               - Asking for clarification on a concept they already discussed in detail
               - Expressing unfamiliarity with a core topic previously covered
               - Forgetting the main focus or structure of the conversation

            3. CONTENT CIRCULARITY: Check for and fix instances where the conversation:
               - Loops back to the exact same point without development
               - Contains nearly identical dialogue exchanges in different segments
               - Restarts the same line of reasoning multiple times

            4. INFORMATION VALUE: Every segment should contribute substantive new insights
               beyond what was already covered. If a segment feels like it's merely
               rehashing previous content, rewrite it to explore new dimensions of the topic.

            TRANSITION STYLE GUIDE:
            - Avoid phrases like "segues into" or "next topic"
            - Connect topics through shared themes or related ideas
            - Use natural conversational bridges like "That reminds me of..." or
              "You know what's interesting about that..."
            - Let one host's insight naturally lead to the next area of discussion

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
