from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.output_parsers import PydanticOutputParser


class SoundEffect(str, Enum):
    TRANSITION = "transition_whoosh"
    HIGHLIGHT = "soft_ding"
    INSIGHT = "lightbulb_moment"
    QUESTION = "thinking_pause"
    DRAMATIC = "dramatic_impact"
    AUDIENCE = "audience_reaction"
    REVEAL = "revelation_chord"

class MusicMood(str, Enum):
    UPBEAT = "upbeat_background"
    THOUGHTFUL = "contemplative_piano"
    INTENSE = "deep_focus"
    LIGHT = "light_ambient"
    OUTRO = "fade_out_theme"
    EXPERT = "professional_strings"
    MYSTERY = "curiosity_builder"

class ListenerQuestion(BaseModel):
    question: str
    asker_name: str
    difficulty_level: int
    relevance_score: float
    related_topics: List[str]

class ExpertInsight(BaseModel):
    expert_name: str
    credentials: str
    key_insight: str
    connection_to_topic: str
    follow_up_points: List[str]
    recommended_resources: List[str]

class PodcastChunk(BaseModel):
    topic: str = Field(description="Main topic of this chunk")
    original_content: str = Field(description="Ignore this field")
    abstraction_level: int = Field(description="Abstraction level from 1-5, where 1 is most abstract/high-level")
    key_points: List[str] = Field(description="Key points covered in this chunk")
    prerequisites: List[str] = Field(description="Topics that should be covered before this chunk")
    follows_from: List[str] = Field(description="Topics this naturally follows from")

class ScriptSegment(BaseModel):
    speaker: str = Field(description="Speaker identifier")
    dialogue: str = Field(description="The actual speech content")
    tone: str = Field(description="Emotional tone of delivery")
    purpose: str = Field(description="Purpose of this segment")
    sound_effect: Optional[SoundEffect] = Field(default=None)
    background_music: Optional[MusicMood] = Field(default=None)
    expert_insight: Optional[ExpertInsight] = Field(default=None)
    listener_question: Optional[ListenerQuestion] = Field(default=None)

class ScriptSegmentList(BaseModel):
    segments: List[ScriptSegment]

class PodcastScript(BaseModel):
    segments: List[ScriptSegment]
    total_speakers: int
    music_cues: int
    sound_effects: int

class PodcastEpisode(BaseModel):
    title: str = Field(description="Catchy, informative title for the episode")
    subtitle: str = Field(description="Brief hook or tagline")
    description: str = Field(description="Engaging episode description for listeners")
    keywords: List[str] = Field(description="SEO and discovery keywords")
    target_audience: str = Field(description="Primary audience for this episode")


class ContentOrganizer:
    def __init__(self, model):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
        self.parser = PydanticOutputParser(pydantic_object=PodcastChunk)
        self.llm = model

    def analyze_chunk_template(self):
        return ChatPromptTemplate([
            ("system", "Analyze this content chunk and structure it for a podcast. Focus on identifying its abstraction level and relationships to other topics. Format using the following instructions: {format_instructions}"),
            ("user", "Content chunk: {chunk}")
        ], partial_variables={"format_instructions": self.parser.get_format_instructions()})

    def process_document(self, document: str) -> List[PodcastChunk]:
        chunks = self.text_splitter.split_text(document)

        analysis_chain = (
                {"chunk": RunnablePassthrough()}
                | self.analyze_chunk_template()
                | self.llm
                | self.parser
        )

        analyzed_chunks = [analysis_chain.invoke(chunk) for chunk in chunks]
        for chunk, original_data in zip(analyzed_chunks, chunks):
            chunk.original_content = original_data
        return self.organize_chunks(analyzed_chunks)

    def organize_chunks(self, chunks: List[PodcastChunk]) -> List[PodcastChunk]:
        sorted_chunks = sorted(chunks, key=lambda x: x.abstraction_level)
        organized_chunks = []
        remaining_chunks = sorted_chunks.copy()

        while remaining_chunks:
            next_chunk = self._find_next_suitable_chunk(organized_chunks, remaining_chunks)
            organized_chunks.append(next_chunk)
            remaining_chunks.remove(next_chunk)

        return organized_chunks

    def _find_next_suitable_chunk(self,
                                  organized: List[PodcastChunk],
                                  remaining: List[PodcastChunk]) -> PodcastChunk:
        covered_topics = {chunk.topic for chunk in organized}

        for chunk in remaining:
            prerequisites_met = all(
                prereq in covered_topics for prereq in chunk.prerequisites
            )
            if prerequisites_met:
                return chunk

        return min(remaining, key=lambda x: x.abstraction_level)


class InteractiveScriptGenerator:
    def __init__(self, model):
        self.llm = model
        self.hosts = ["Alex", "Jordan"]  # @TODO: Customize via config
        self.dialogue_parser = PydanticOutputParser(pydantic_object=ScriptSegmentList)
        self.segments = []
        self.setup_templates()

    def setup_templates(self):
        # @TODO: Extract the voice and host information out
        self.dialogue_template = ChatPromptTemplate([
            ("system", """
            "Create engaging podcast dialogue that:
            - Uses conversational language
            - Includes natural host interactions
            - Creates memorable analogies
            - Builds curiosity
            - Sounds natural with filler words like umm, hmm, etc.
            
            The podcast is driven by:
            - Alex, who is curious, funny and inquisitive
            - Jordan, who is more reserved and analytical with a knack of making connections and explaining complex ideas in simple terms. They are a master at providing accurate analogies and examples. They love a good joke, once in a while 
            
            REMEMBER THIS IS JUST A SEGMENT OF A PODCAST, SO NO NEED FOR OPENING OR CLOSING REMARKS IN THE DIALOGUE.
            REMEMBER THAT THERE ARE MULTIPLE SEGMENTS IN THIS EPISODE, SO STEER AWAY FROM PHRASES LIKE "Good question", "Good analogy", etc. THAT CAN GET REPETITIVE ACROSS MULTIPLE SEGMENTS.
               
            Keep the tone casual but informative.
            Keep responses focused and clear.
            Format as: {format_instructions}
            """),
            ("user", """
            Source content: {content_chunk}
            Topic: {topic}
            Key Points: {key_points}
            Previous Discussion: {previous_points}
            Purpose: {purpose}""")
        ], partial_variables={"format_instructions": self.dialogue_parser.get_format_instructions()})

        self.expert_template = ChatPromptTemplate.from_messages([
            ("system", "Create expert commentary that adds depth while remaining accessible."),
            ("user", "Topic: {topic}\nExpertise: {expertise}\nKey Points: {key_points}")
        ])

        self.question_template = ChatPromptTemplate.from_messages([
            ("system", "Generate relevant listener questions that drive discussion forward."),
            ("user", "Topic: {topic}\nCurrent Understanding Level: {level}")
        ])

    def generate_episode_metadata(self, organized_chunks: List[PodcastChunk]) -> PodcastEpisode:
        parser = PydanticOutputParser(pydantic_object=PodcastEpisode)
        metadata_template = ChatPromptTemplate([
            ("system", "Create engaging podcast metadata that captures the essence of the episode and attracts target listeners. Format as: {format_instructions}"),
            ("user", """Main topics: {topics}
            Key points: {points}
            """)
        ], partial_variables={"format_instructions": parser.get_format_instructions()})

        metadata_chain = metadata_template | self.llm | parser
        return metadata_chain.invoke({"topics": [chunk.topic for chunk in organized_chunks[:3]], "points": [point for chunk in organized_chunks for point in chunk.key_points[:2]]})

    def generate_script(self, organized_chunks: List[PodcastChunk]) -> PodcastScript:
        self.segments = []

        self.episode_metadata = self.generate_episode_metadata(organized_chunks)

        # Add opening sequence
        self.segments.extend(self._generate_opening_sequence(self.episode_metadata))

        # Process main content
        for i, chunk in tqdm(enumerate(organized_chunks), desc="Generating script"):
            # Add transition if needed
            if i > 0:
                self.segments.extend(self._generate_transition(organized_chunks[i-1], chunk))

            # Generate main content segments
            self.segments.extend(self._generate_content_segment(chunk))

            # Add interactive elements
            # if self._should_add_listener_question(i):
            #     self.segments.extend(self.generate_listener_segment(chunk))
            #
            # if self._should_add_expert_insight(chunk):
            #     self.segments.extend(self.generate_expert_segment(chunk))

        # Add closing sequence
        self.segments.extend(self._generate_closing_sequence(organized_chunks))

        return PodcastScript(
            segments=self.segments,
            total_speakers=len(set(segment.speaker for segment in self.segments)),
            music_cues=sum(1 for segment in self.segments if segment.background_music),
            sound_effects=sum(1 for segment in self.segments if segment.sound_effect)
        )

    def _generate_transition(self, previous_chunk: PodcastChunk, next_chunk: PodcastChunk) -> List[ScriptSegment]:
        transition_prompt = {
            "previous_topic": previous_chunk.topic,
            "next_topic": next_chunk.topic,
            "connection_points": list(set(previous_chunk.key_points) & set(next_chunk.follows_from))[:2],
            "purpose": "create natural bridge between topics"
        }

        transition_template = ChatPromptTemplate([
            ("system", """Create a quick, natural bridge between topics using 1-2 lines of dialogue. 
            Make an interesting connection or observation that leads perfectly into the next topic.
            Keep it snappy and engaging.
            
            The podcast is driven by:
            - Alex, who is curious and inquisitive
            - Jordan, who is more reserved and analytical with a knack of making connections and explaining complex ideas in simple terms. He is a master at providing accurate analogies and examples.
            
            Keep it concise and engaging.
            Format response as: {format_instructions}
            """),
            ("user", """Previous Topic: {previous_topic}
            Next Topic: {next_topic}
            Connection Points: {connection_points}
            Purpose: {purpose}""")
        ], partial_variables={"format_instructions": self.dialogue_parser.get_format_instructions()})

        chain = transition_template | self.llm | self.dialogue_parser

        return [
            ScriptSegment(
                speaker="PRODUCTION",
                dialogue="[Transition ambience]",
                tone="neutral",
                purpose="transition",
                sound_effect=SoundEffect.TRANSITION,
                background_music=MusicMood.LIGHT
            ),
            *chain.invoke(transition_prompt).segments
        ]

    def _generate_content_segment(self, chunk: PodcastChunk) -> List[ScriptSegment]:
        return self._generate_key_points_discussion(chunk)

    def _should_add_listener_question(self, index: int) -> bool:
        return index % 2 == 0  # Add questions every other segment

    def _should_add_expert_insight(self, chunk: PodcastChunk) -> bool:
        return chunk.abstraction_level >= 3  # Add expert insights for complex topics

    def _generate_closing_sequence(self, chunks: List[PodcastChunk]) -> List[ScriptSegment]:
        closing_prompt = {
            "content_chunk": str([chunk.topic for chunk in chunks]),
            "key_points": [point for chunk in chunks[:3] for point in chunk.key_points],
            "episode_title": self.episode_metadata.title,
            "purpose": "wrap up key insights and create anticipation",
            "previous_points": self._get_previous_points()
        }

        closing_template = ChatPromptTemplate([
            ("system", """Create an engaging podcast closing that:
            - Synthesizes key insights from the discussion
            - Makes unexpected connections between topics
            - Leaves listeners with a thought-provoking takeaway
            - Creates anticipation for future exploration
            - Only focus on dialogue, don't add any music or effects segments
            Keep it punchy and memorable. 
            
            The podcast is driven by:
            - Alex, who is curious and inquisitive
            - Jordan, who is more reserved and analytical with a knack of making connections and explaining complex ideas in simple terms. He is a master at providing accurate analogies and examples.
            
            
            Format response as: {format_instructions}
            """),
            ("user", """Episode Title: {episode_title}
            Topics Covered: {content_chunk}
            Key Insights: {key_points}
            Previous Discussion: {previous_points}
            Purpose: {purpose}""")
        ], partial_variables={"format_instructions": self.dialogue_parser.get_format_instructions()})

        chain = closing_template | self.llm | self.dialogue_parser
        closing_dialogue = chain.invoke(closing_prompt)

        return [
            ScriptSegment(
                speaker="PRODUCTION",
                dialogue="[Thoughtful closing theme]",
                tone="reflective",
                purpose="outro_begin",
                background_music=MusicMood.THOUGHTFUL,
                sound_effect=SoundEffect.INSIGHT
            ),
            *closing_dialogue.segments,
            ScriptSegment(
                speaker="PRODUCTION",
                dialogue="[Theme music swells]",
                tone="professional",
                purpose="outro_end",
                background_music=MusicMood.OUTRO
            )
        ]

    def generate_listener_segment(self, chunk: PodcastChunk) -> List[ScriptSegment]:
        question = self._generate_relevant_question(chunk)
        return [
            ScriptSegment(
                speaker="PRODUCTION",
                dialogue="[Question submission chime]",
                sound_effect=SoundEffect.QUESTION,
                purpose="transition"
            ),
            ScriptSegment(
                speaker=self.hosts[0],
                dialogue=f"We have an interesting question from {question.asker_name}",
                tone="welcoming",
                purpose="introduce_question"
            ),
            ScriptSegment(
                speaker="Listener",
                dialogue=question.question,
                tone="curious",
                purpose="ask_question"
            ),
            *self._generate_question_response(question, chunk)
        ]

    def _generate_opening_sequence(self, metadata) -> List[ScriptSegment]:
        opening_prompt = {
            "content_chunk": metadata.description,
            "previous_points": None,
            "topic": metadata.title,
            "key_points": [metadata.subtitle],
            "tone": "enthusiastic",
            "purpose": "podcast introduction and engage audience"
        }

        chain = self.dialogue_template | self.llm | self.dialogue_parser
        opening_dialogue = chain.invoke(opening_prompt)

        return [
            ScriptSegment(
                speaker="PRODUCTION",
                dialogue="[Theme music begins]",
                tone="professional",
                purpose="intro",
                background_music=MusicMood.UPBEAT
            ),
            *opening_dialogue.segments
        ]

    def _generate_key_points_discussion(self, chunk: PodcastChunk) -> List[ScriptSegment]:
        dialogue_prompt = {
            "content_chunk": chunk.original_content,
            "topic": chunk.topic,
            "key_points": chunk.key_points,
            "previous_points": self._get_previous_points(),
            "purpose": "explore and connect concepts",
        }

        chain = self.dialogue_template | self.llm | self.dialogue_parser
        return chain.invoke(dialogue_prompt).segments


    def _get_previous_points(self) -> List[str]:
        return [segment.dialogue for segment in self.segments[-3:] if segment.speaker in self.hosts]

    def _determine_tone(self, dialogue: str) -> str:
        # Implement tone analysis based on dialogue content
        tones = ["curious", "excited", "thoughtful", "analytical", "enthusiastic"]
        return tones[hash(dialogue) % len(tones)]

    def _generate_engagement_effect(self) -> ScriptSegment:
        effects = [
            (SoundEffect.INSIGHT, MusicMood.THOUGHTFUL),
            (SoundEffect.HIGHLIGHT, MusicMood.LIGHT),
            (SoundEffect.DRAMATIC, MusicMood.INTENSE)
        ]
        effect, music = effects[len(effects) % len(effects)]

        return ScriptSegment(
            speaker="PRODUCTION",
            dialogue="[Effect]",
            tone="neutral",
            purpose="engagement",
            sound_effect=effect,
            background_music=music
        )

def generate_podcast(model, document: str):
    # Initialize components
    organizer = ContentOrganizer(model)
    generator = InteractiveScriptGenerator(model)

    # Process content
    organized_chunks = organizer.process_document(document)

    # Generate script
    script = generator.generate_script(organized_chunks)
    return script
