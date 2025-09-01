from llama_cpp import Llama

llm = Llama(
    model_path="/Users/borjasanchez/llama.cpp/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=6,
    n_gpu_layers=-1  # Uses your Mac's GPU (Metal)
)

output = llm("Where is Tokyo?", max_tokens=20)

print(output["choices"][0]["text"].strip())