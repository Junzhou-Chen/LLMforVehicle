from langchain.llms.llamacpp import LlamaCpp
# from llama_cpp import Llama
model_path = r'G:\projects\llama.cpp\models\13B\ggml-model-q8_0.gguf'
message = "Write me a poem'"
llm = LlamaCpp(model_path=model_path,
               verbose=False,
               temperature=0.2,
               n_gpu_layers=25,
               n_ctx=4096,
               max_tokens=1024,
               n_batch=16,
               repeat_penalty=1.1)

# print(llm(message))

for s in llm.stream("write me a poem"):
    print(s, end="", flush=True)
