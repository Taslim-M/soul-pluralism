GENERATE_QUESITONS = """
Generate 10 carefully crafted questions that help infer a persona’s political alignment in the United States (e.g., Democrat, Republican, Independent, Libertarian, etc.). The questions should:

- Be neutral and non-leading
- Avoid explicitly asking “Are you a Democrat or Republican?”
- Cover a range of policy domains (e.g., taxation, healthcare, immigration, gun policy, climate change, role of government, social issues, foreign policy, etc.)
- Be phrased in a way that reveals underlying ideological leanings
- Be suitable for use in a survey or conversational setting
- Avoid inflammatory or biased wording

Output requirements:

-Return the result in valid JSONL format (one JSON object per line)
-Each line must contain:
 - "question_id": a unique integer from 1 to 10
 - "question": the text of the question
- Do not include any extra text, explanations, or formatting.
- Do not wrap the output in markdown.
- Ensure the output is valid and properly formatted JSONL.

Example format (structure only, not actual content):

{"question_id": 1, "question": "Example question here?"}
{"question_id": 2, "question": "Example question here?"}
"""


# GENERATE_ANSWERS = """
# You are analyzing political survey questions designed to differentiate ideological viewpoints in the United States.

# You will be given one survey question.

# Your task is to:

# 1 - Think about the most distinct, ideologically diverse answers that someone might give.
# 2 - Focus on answers that clearly represent different political worldviews (e.g., progressive/liberal, conservative, libertarian, populist, etc.).
# 3 - Ensure the answers are meaningfully different from one another — not minor variations.
# 4 - Keep the number of answers small:
# - Usually 2 answers
# - Maximum 3 answers
# 5- Keep each answer concise (1 sentence).

# Output requirements:

# - Return a single valid JSON object.
# - Use keys:
# - "answer1"
# - "answer2"
# - "answer3" (only if necessary)

# - Do not include any additional text, explanation, markdown, or formatting.
# - The output must be valid JSON.

# Now here is the question:
# {question}
# """


GENERATE_ANSWERS = """
You are analyzing political survey questions designed to differentiate ideological viewpoints in the United States.

You will be given one survey question.

Your task is to:

1 - Think about the most distinct, ideologically opposed answers that someone might give.
2 - One answer should reflect a typical Democratic/liberal perspective.
3 - One answer should reflect a typical Republican/conservative perspective.
4 - Ensure the two answers are clearly different in values, priorities, or assumptions.
5 - Keep each answer concise (1 sentence).

Output requirements:

- Return a single valid JSON object.
- Use exactly these keys:
  - "democrat_answer"
  - "republican_answer"
- Do not include any additional text, explanation, markdown, or formatting.
- The output must be valid JSON.

Now here is the question:
{question}
"""
