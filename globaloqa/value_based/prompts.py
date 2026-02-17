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
