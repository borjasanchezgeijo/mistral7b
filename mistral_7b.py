from llama_cpp import Llama

llm = Llama(
    model_path="/Users/borjasanchez/llama.cpp/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=6,
    n_gpu_layers=-1,
    verbose=False
)

prompt = input("How can I help you today?").strip()

output = llm(prompt, max_tokens=40)

print("\033[91m" + output["choices"][0]["text"].strip() + "\033[0m")
