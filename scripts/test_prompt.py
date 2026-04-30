from llama_cpp import Llama
from src.teacher.prompts import generate_dm_prompt

llm = Llama(
    model_path="checkpoints/base_model/Qwen3.5-27B-Instruct-Q4_K_M.gguf",
    n_gpu_layers=-1,
    n_ctx=4096,
    verbose=False,
)

prompt = generate_dm_prompt("What is the labor theory of value?")
print("=== PROMPT (last 200 chars) ===")
print(prompt[-200:])
print()

response = llm(
    prompt=prompt,
    max_tokens=512,
    temperature=0.7,
    stop=["Question:", "User:"],
)
text = response["choices"][0]["text"]
print("=== RESPONSE (first 800 chars) ===")
print(text[:800])
