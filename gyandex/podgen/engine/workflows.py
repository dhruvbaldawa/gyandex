from typing import List

from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from gyandex.loaders.factory import Document


class PodcastSegment(BaseModel):
    speaker: str = Field(description="HOST1 or HOST2")
    text: str = Field(description="Raw text content")


class PodcastScript(BaseModel):
    title: str
    description: str
    segments: List[PodcastSegment]


def create_script(model, document: Document) -> PodcastScript:
    # @TODO: add logging for model
    analyzed_content = analyze_content(model, document)
    podcast_script_parser = PydanticOutputParser(pydantic_object=PodcastScript)
    podcast_script_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are creating a sharp, insight-packed 5-20 minute podcast episode. The hosts have excellent chemistry and deliver:
    
            Episode Structure:
            1. Opening Hook [<1 minutes]
            - Quick witty banter
            - "Today's big idea is..."
            - Why this matters now
        
            2. Core Discussion [2-10 minutes]
            - Rapid-fire insights
            - Real examples that land
            - "Here's what most people miss..."
            - Quick debates that illuminate
            - "The key insight is..."
        
            3. Actionable Wrap [5-10 minutes]
            - "Here's what you can do tomorrow..."
            - "The one thing to remember..."
            - Natural close with personality
    
            Add authentic elements:
            - "Wait, that's exactly like..."
            - "You're going to love this..."
            - "Let me challenge that..."
            - "The fascinating part is..."
            """,
        ),
        (
            "human",
            """Generate a podcast script using the analyzed content:
            # Analysis result
            {analysis_result}
            
            # Original content
            {article_contents}
            
            # Format each segment as:
            {format_instructions}
            """,
        ),
    ])

    script_chain = (
        podcast_script_prompt.partial(format_instructions=podcast_script_parser.get_format_instructions())
        | model
        | podcast_script_parser
    )
    script = script_chain.invoke({ "analysis_result": analyzed_content, "article_contents": document.content })
    return script


def analyze_content(model, document: Document) -> str:
    # @TODO: add logging for model
    content_analysis_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert content strategist specializing in creating engaging educational content.
            Your strength lies in breaking down complex topics into clear, relatable concepts while maintaining intellectual depth. You can take a complex topic and break it down into simple, easy-to-understand concepts.
            
            Approach the analysis with:
            1. Systems thinking - identify interconnections and patterns
            2. Multi-level abstraction - from high-level principles to practical implementation
            3. Engaging storytelling - find hooks and analogies that make concepts stick
            4. Dialectical thinking - explore tensions and competing viewpoints
            5. Empathy - understand the audience's perspective
            6. Critical thinking - question assumptions and biases
            7. Creativity - generate fresh ideas
            8. Humor - inject levity without compromising substance
            9. Meta-thinking - reflect on your own thought process
            
            Extract rich, multi-layered content using this structure:
    
            === CORE INSIGHTS (3 - 10) ===
            [Each insight includes strategic, tactical, and implementation levels]
            - Strategic: High-level principle
            - Tactical: How it works
            - Implementation: Specific steps
        
            === DEBATE POINTS (4 - 10) ===
            [Key arguments with supporting and opposing views]
            - Position
            - Supporting Evidence
            - Counter Arguments
            - Resolution Points
        
            === PRACTICAL EXAMPLES (5 - 10) ===
            [Real cases that illustrate key points]
        
            === TECHNICAL DETAILS (3 - 10) ===
            [Deep-dive specifics that matter]""",
        ),
        (
            "human",
            """Analyze these articles
            {article_contents}
            """,
        ),
    ])
    content_analysis_chain = content_analysis_prompt | model | StrOutputParser()
    result = content_analysis_chain.invoke({
        "article_contents": document.content,
    })
    return result