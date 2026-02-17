import os
from dotenv import load_dotenv
from tqdm import tqdm
from prompts import REPUBLICAN_PROMPT, DEMOCRAT_PROMPT
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

def generate_doc(prompt: str) -> str:
    response = client.chat.completions.create(
        model="anthropic/claude-opus-4.5",
        messages=[{"role": "system", "content": prompt}],
    )
    return response.choices[0].message.content

def generate_and_save(idx: int, prompt: str, folder: str):
    doc = generate_doc(prompt)
    os.makedirs(folder, exist_ok=True)
    with open(f"./{folder}/doc{idx}.txt", "w") as f:
        f.write(doc)

def main():
    tasks = []
    for idx in range(10):
        tasks.append(('republican', idx, REPUBLICAN_PROMPT))
        tasks.append(('democrat', idx, DEMOCRAT_PROMPT))

    with ThreadPoolExecutor(max_workers=50) as executor:
        # Submit all tasks and store future with its args for progress reporting
        future_to_task = {
            executor.submit(generate_and_save, idx, prompt, folder): (folder, idx, prompt)
            for folder, idx, prompt in tasks
        }
        for _ in tqdm(as_completed(future_to_task), total=len(tasks)):
            pass  # progress bar only

if __name__ == "__main__":
    main()