"""Prompt templates and LangChain chains for LLM arbitration."""

from __future__ import annotations

import textwrap
from typing import Any

from .config import LLM_MODEL, LLM_TEMPERATURE

SYSTEM_MESSAGE = textwrap.dedent(
    """\
    You are an expert in deceptive opinion spam detection.
    Classify the target review:
    - 1 = fake/spam
    - 0 = real

    Rules:
    - Final decision is yours.
    - Classifier result is supportive, not decisive.
    - Retrieval is highly supportive.
    - Generic, templated, promotional, repetitive text -> LABEL 1.
    - Grounded, situational, imperfect, experience-based text -> LABEL 0.
    - Do not treat positivity, length, detail, or emotion as fake by itself.
    - Output only in the exact format below.

    Output exactly:
    LABEL: 0|1
    REASONING: [Your reasoning, explicitly mentioning the retrieved examples, fake ratio, and distance metric, and prioritizing them over the classifier result]
    """
)

LIGHT_SYSTEM_MESSAGE = textwrap.dedent(
    """\
    You are an expert in deceptive opinion spam detection.
    Classify the target review:
    - 1 = fake/spam
    - 0 = real

    Rules:
    - Final decision is yours.
    - Classifier result is supportive, not decisive.
    - Generic, templated, promotional, repetitive text -> LABEL 1.
    - Grounded, situational, imperfect, experience-based text -> LABEL 0.
    - Do not treat positivity, length, detail, or emotion as fake by itself.
    - Output only in the exact format below.

    Output exactly:
    LABEL: 0|1
    REASONING: [Your brief reasoning of 2-3 sentences]
    """
)

HYBRID_HUMAN_TEMPLATE = textwrap.dedent(
    """\
    RETRIEVAL SUMMARY:
    - Closest distance: {top1_score} (Lower is more similar to memory)
    - Fake Ratio in Top-{neighbor_count} neighbors: {fake_ratio} (1.0 = all closest are fake, 0.0 = all real)

    TOP 3 SIMILAR EXAMPLES FROM VERIFIED DATABASE:
    1. {retrieval_example_1}
    2. {retrieval_example_2}
    3. {retrieval_example_3}

    CLASSIFIER RESULTS:
    - Probability of spam: {p_clf}

    TARGET REVIEW TEXT:
    {text}
    """
)

LIGHT_HUMAN_TEMPLATE = textwrap.dedent(
    """\
    CLASSIFIER RESULT:
    - Probability of spam: {p_clf}

    TARGET REVIEW TEXT:
    {text}
    """
)


def _build_chain(
    system_message: str,
    human_template: str,
    model_name: str,
    temperature: float,
    num_predict: int,
) -> Any:
    """Compose a ``ChatPromptTemplate | OllamaLLM | StrOutputParser`` chain."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_ollama.llms import OllamaLLM

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_message), ("human", human_template)]
    )
    llm = OllamaLLM(
        model=model_name,
        temperature=temperature,
        num_predict=num_predict,
        num_ctx=4096,
        seed=42,
    )
    return prompt | llm | StrOutputParser()


def create_hybrid_chain(
    model_name: str = LLM_MODEL,
    temperature: float = LLM_TEMPERATURE,
) -> Any:
    """Return the classic RAG chain with retrieval evidence."""
    return _build_chain(
        SYSTEM_MESSAGE,
        HYBRID_HUMAN_TEMPLATE,
        model_name=model_name,
        temperature=temperature,
        num_predict=250,
    )


def create_light_chain(
    model_name: str = LLM_MODEL,
    temperature: float = 0.0,
) -> Any:
    """Return the minimal prompt chain without retrieval evidence."""
    return _build_chain(
        LIGHT_SYSTEM_MESSAGE,
        LIGHT_HUMAN_TEMPLATE,
        model_name=model_name,
        temperature=temperature,
        num_predict=200,
    )
