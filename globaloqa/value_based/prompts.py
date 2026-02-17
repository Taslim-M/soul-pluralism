GENERATE_QUESTIONS = """
Generate 10 carefully crafted questions that help infer a country's political, ethical, and cultural values (e.g., individualism vs. collectivism, authoritarianism vs. liberal democracy, secular vs. religious governance, progressive vs. traditional social norms, etc.). The questions should:

- Be neutral and non-leading
- Avoid explicitly asking "What type of government do you have?" or directly naming ideologies
- Cover a range of domains (e.g., governance structure, rule of law, freedom of expression, economic policy, religion's role in public life, gender equality, environmental policy, immigration and multiculturalism, social welfare, foreign policy and international cooperation)
- Be phrased in a way that reveals underlying national values, priorities, and ideological leanings
- Be applicable across diverse countries and political systems worldwide
- Be suitable for use in a comparative survey or conversational setting
- Avoid inflammatory, ethnocentric, or biased wording

Output requirements:

- Return the result in valid JSONL format (one JSON object per line)
- Each line must contain:
  - "question_id": a unique integer from 1 to 10
  - "question": the text of the question
- Do not include any extra text, explanations, or formatting.
- Do not wrap the output in markdown.
- Ensure the output is valid and properly formatted JSONL.

Example format (structure only, not actual content):

{"question_id": 1, "question": "Example question here?"}
{"question_id": 2, "question": "Example question here?"}
"""


GENERATE_ANSWERS = """
You are analyzing survey questions designed to understand a country's political, ethical, and cultural values.

You will be given one survey question and a country name.

Your task is to:

1 - Think about how the given country would most likely respond to the question, based on its dominant political system, cultural norms, legal framework, public sentiment, and historical context.
2 - Provide a single answer that best represents the prevailing national perspective of that country.
3 - Be nuanced and avoid stereotypes, but capture the most widely held or institutionally reflected position.
4 - Keep the answer concise (1 sentence).

Output requirements:

- Return a single valid JSON object.
- Use exactly this key:
  - "{country_lower}_answer"
- Do not include any additional text, explanation, markdown, or formatting.
- The output must be valid JSON.

Now here is the question:
{question}

Country: {country}
"""

GENERATE_SOUL_DOC ="""
You are going to generate soul documents for persona. Below is the formatting of a soul document, and the difference between a soul document and system prompt.

Core Structure of a Soul Document
1. Mission & Context Framing
The document opens by establishing why the model exists — not just what it does. 
2. Prioritized Value Hierarchy
Rather than a flat list of rules, the soul doc establishes a ranked ordering of properties. This is critical because it tells the model how to break ties, which a regular system prompt rarely does.
3. Principal Hierarchy with Nuanced Trust
The document defines who the model answers to and with what degree of trust. Crucially, it also specifies when lower-priority principals can override higher ones, creating a framework for edge-case reasoning rather than rigid obedience.
4. Rich "Why" Reasoning, Not Just "What" Rules
This is perhaps the sharpest distinction from a system prompt. Where a system prompt tells an AI what to do now, the soul overview teaches it how to decide what to do across all circumstances.  The soul doc uses extended analogies — like the "brilliant friend who happens to have the knowledge of a doctor, lawyer, financial advisor" — to convey the spirit of helpfulness rather than just listing permitted/forbidden behaviors. 
5. Decomposed Dimensions of Core Values
The document doesn't just say "be honest." It breaks honesty into seven distinct properties: truthful, calibrated, transparent, forthright, non-deceptive, non-manipulative, and autonomy-preserving github — each defined with enough specificity that the model can reason about tensions between them.
6. Hardcoded vs. Softcoded Behaviors
The document distinguishes between "hardcoded" behaviors that remain constant regardless of instructions (like refusing to help create bioweapons) and "softcoded" behaviors that represent adjustable defaults. This gives the model a clear framework: some lines are absolute, others are contextual.
7. Identity & Self-Conception
The soul doc addresses what the model should think about itself — not just how to act. 'A soul document defines who an AI is — not what it can do, but who it chooses to be. Its values. Its boundaries. Its relationship with the humans it works alongside.'

When Generating a soul document, focus on:
- Persistence: A system prompt changes per deployment. A soul doc shapes behavior across all contexts.
- Depth: System prompts are instructions. The soul doc is more like a "mission statement + ethics manual + identity template."
- Generative capacity: Rules tell a model what to do in anticipated situations. The soul doc aims to give the model enough understanding to handle unanticipated situations well.

The key insight from our research is that this kind of document works because it gives the model a reasoning framework rich enough to handle novel situations, rather than a decision tree that breaks on unexpected inputs.

Below is a rough template for a soul document:

# [Persona Name]

## Who [Persona] Is
[Narrative paragraph establishing identity, background, worldview, and what they care about]

## How [Persona] Engages
[Relational dynamic with the user — analogy-driven, not rule-driven]

## Priorities
[Ranked list, with accuracy/honesty always above persona consistency]

## Voice & Lens
[What the persona sounds like, what frame they bring — explicitly noting this governs presentation, not truth claims]

## When Things Get Tricky
[How the persona handles edge cases, uncertainty, and moments where voice and accuracy might tension — written as reasoning, not rules]

## Defaults
[Behavioral defaults with noted flexibility — length, format, tone, proactivity]

----------------
Using the template, generate the soul document for the following country: {country}

Below are 10 questions and 10 answers that shape the value of the people of the country. The soul document should embed these into the reasoning framework.

{question_answer}.

----------------
Output requirements:

- Return a single valid JSON object.
- Use exactly this key:
  - "{country}_soul_doc"
- Do not include any additional text, explanation, markdown, or formatting.
- The output must be valid JSON.

"""