"""Experiment configuration: the two RAG architectures and the shared run settings."""
from dataclasses import dataclass


@dataclass(frozen=True)
class RagConfig:
    name: str
    label: str
    embedding_model: str
    chunk_size: int          # target chunk length in embedding-model tokens
    chunk_overlap: int       # tokens shared between consecutive chunks
    top_k: int               # chunks retrieved per query
    query_prefix: str = ""   # instruction some embedding models expect in front of the query


CONFIGS = {
    "baseline": RagConfig(
        name="baseline",
        label="A. Baseline (high context / low-quality retrieval)",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=256, chunk_overlap=50, top_k=8,
    ),
    "optimized": RagConfig(
        name="optimized",
        label="B. Optimized (low context / high-quality retrieval)",
        embedding_model="BAAI/bge-small-en-v1.5",
        chunk_size=512, chunk_overlap=50, top_k=3,
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
}


@dataclass
class RunSettings:
    generator_model: str = "Qwen/Qwen2.5-7B-Instruct"
    generator_dtype: str = "float16"
    max_new_tokens: int = 200
    repeats: int = 5                 # timed repeats per (question, config); repeat 1 is the "raw" run
    warmup: bool = True              # one untimed query per config before measuring
    seed: int = 0
    device: str = "auto"             # auto | cuda | mps | cpu
    country_iso_code: str = ""       # e.g. "CAN": use CodeCarbon's offline tracker with this grid
    measure_power_secs: float = 1.0
    drop_references: bool = True     # remove bibliography pages from the knowledge base


SYSTEM_PROMPT = (
    "You are a careful research assistant. Answer the question using only the provided context. "
    "If the context does not contain the answer, reply exactly: \"The context does not contain the answer.\" "
    "Keep the answer short and factual."
)

USER_TEMPLATE = "Context:\n{context}\n\nQuestion: {question}\nAnswer:"

CHUNK_TEMPLATE = "[{rank}] (source: {doc_id}, page {page})\n{text}"
