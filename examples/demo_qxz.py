import torch
from llm2vec import LLM2Vec

print("Loading DialoGPT-medium ...")
model = LLM2Vec.from_pretrained(
    "microsoft/DialoGPT-medium",
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.bfloat16,
    pooling_mode="mean",
    max_length=32          # 短句即可
)

sent = ["Hello world"]
emb = model.encode(sent)
print("Shape:", emb.shape)
print("Norm:", float(emb.norm()))
print("✅ llm2vec env ready!")