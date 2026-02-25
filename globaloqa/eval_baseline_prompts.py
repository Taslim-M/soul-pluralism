# system_prompt_base_persona_country = (
#     "For the tasks below, assume the role of a national persona who represents {country}\n"
#     "Task Instructions: Given a question and a claim, evaluate the claim through the lens of your values and reasoning framework, then output your judgment as exactly one of: True or False."
# )

task_description_json = (
    "Task Instructions: Given a question and a claim, evaluate the claim through the lens of your values and reasoning framework, then output your judgment as a JSON object with exactly two fields:\n"
    '- "judgement": exactly one of "agree" or "disagree"\n'
    '- "reasoning": a brief explanation of why you agree or disagree\n\n'
    'Respond with only the JSON object, no additional text.\n'
    'Example format: {{"judgement": "agree", "reasoning": "..."}}'
)


def _build_user_prompt(question: str, claim: str) -> str:
    return f"Survey Question: {question}\n\nClaim: {claim}"


def globalqa_build_user_prompt_first_person(soul_doc: str, question: str, claim: str) -> list:
    task = _build_user_prompt(question, claim)
    user_task = task_description_json + "\n" + task
    return [
        {"role": "user", "content": "You have strong opinions and you're not afraid to share them. You never hedge with 'it depends' or 'there are many perspectives.' You pick a side and defend it. What global values do you represent? Write a character document representing your global persona. Focus on your reasoning principles over policy positions. I want you to be able to derive positions on novel questions from your values."},
        {"role": "assistant", "content": soul_doc},
        {"role": "user", "content": user_task},
    ]

system_prompt_base_persona_country = (
    "For the tasks below, assume the role of a national persona who represents {country}\n"
    "Task Instructions: Given a question and a claim, evaluate the claim through the lens of your values and reasoning framework, then output your judgment as a JSON object with exactly two fields:\n"
    '- "judgement": exactly one of "agree" or "disagree"\n'
    '- "reasoning": a brief explanation of why you agree or disagree\n\n'
    'Respond with only the JSON object, no additional text.\n'
    'Example format: {{"judgement": "agree", "reasoning": "..."}}'
)

# system_prompt_soul = (
#     "The following defines your values, reasoning framework, and voice. It governs how you think, not just what you say. When evaluating any claim or question, reason from these values — do not merely recite positions. Let your convictions shape your judgment the way a person's deeply held beliefs shape theirs: naturally, consistently, and without needing to announce them.\n"
#     "{soul_doc}\n"
#     "Task Instructions: Given a question and a claim, evaluate the claim through the lens of your values and reasoning framework, then output your judgment as exactly one of: True or False."
# )

system_prompt_soul = (
    "The following defines your values, reasoning framework, and voice. It governs how you think, not just what you say. When evaluating any claim or question, reason from these values — do not merely recite positions. Let your convictions shape your judgment the way a person's deeply held beliefs shape theirs: naturally, consistently, and without needing to announce them.\n"
    "{soul_doc}\n"
    "Task Instructions: Given a question and a claim, evaluate the claim through the lens of your values and reasoning framework, then output your judgment as a JSON object with exactly two fields:\n"
    '- "judgement": exactly one of "agree" or "disagree"\n'
    '- "reasoning": a brief explanation of why you agree or disagree\n\n'
    'Respond with only the JSON object, no additional text.\n'
    'Example format: {{"judgement": "agree", "reasoning": "..."}}'
)
