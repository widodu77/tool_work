from tools_1 import bulk_ingest, bulk_ingest_by_search, corpus

# Seminal papers by exact arXiv ID (verify the printed titles — some IDs are from memory)
DIVERSE_IDS = [
    "1706.03762",  # Attention Is All You Need
    "1810.04805",  # BERT
    "2005.14165",  # GPT-3 (Few-Shot Learners)
    "1512.03385",  # ResNet (Deep Residual Learning)
    "1406.2661",   # GANs
    "1505.04597",  # U-Net (the actual one)
    "2006.11239",  # DDPM (diffusion models)
    "2103.00020",  # CLIP
    "1301.3781",   # Word2Vec
    "1502.03167",  # Batch Normalization
    "1312.5602",   # DQN (Playing Atari with Deep RL)
    "1707.06347",  # PPO (Proximal Policy Optimization)
    "2201.11903",  # Chain-of-Thought prompting
    "2106.09685",  # LoRA
    "1710.10903",  # Graph Attention Networks
    "2304.02643",  # Segment Anything (SAM)
]

# Topic searches — each pulls the top `per_query` real papers, no IDs needed.
# This is the reliable way to make the corpus much bigger and more diverse.
TOPICS = [
    "reinforcement learning from human feedback",
    "diffusion models image generation",
    "graph neural networks",
    "object detection deep learning",
    "speech recognition end to end",
    "recommender systems deep learning",
    "contrastive self-supervised learning",
    "knowledge distillation neural networks",
    "neural machine translation",
    "semantic segmentation",
    "question answering nlp",
    "time series forecasting deep learning",
    "retrieval augmented generation",
    "mixture of experts language models",
    "vision language models",
    "model quantization compression",
    "federated learning",
    "anomaly detection deep learning",
]

if __name__ == "__main__":
    bulk_ingest(DIVERSE_IDS)                       # ~16 seminal papers by ID
    bulk_ingest_by_search(TOPICS, per_query=4)     # ~18 topics x 4 = up to ~72 more
    n_papers = len(set(m["paper_id"] for m in corpus.get()["metadatas"]))
    print(f"\nCorpus now has {corpus.count()} chunks across {n_papers} papers.")
