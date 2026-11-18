"""Token-based sliding-window chunking, measured with the embedding model's own tokenizer."""
from dataclasses import asdict, dataclass


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    index: int
    text: str
    n_tokens: int            # embedding-tokenizer tokens
    start_char: int
    end_char: int
    page_start: int
    page_end: int

    def to_dict(self):
        return asdict(self)


def chunk_documents(docs, tokenizer, chunk_tokens, overlap, prefix):
    """Split each document into windows of `chunk_tokens` tokens that overlap by `overlap` tokens.

    Chunks never cross document boundaries. Offsets map every token back to the original text,
    so each chunk is an exact substring of the cleaned document.
    """
    if overlap >= chunk_tokens:
        raise ValueError("overlap must be smaller than the chunk size")
    step = chunk_tokens - overlap
    chunks = []
    for doc in docs:
        enc = tokenizer(doc.text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
        offsets = enc["offset_mapping"]
        n = len(offsets)
        start, i = 0, 0
        while start < n:
            end = min(start + chunk_tokens, n)
            c0, c1 = offsets[start][0], offsets[end - 1][1]
            chunks.append(Chunk(
                chunk_id=f"{prefix}-{doc.doc_id}-{i:04d}", doc_id=doc.doc_id, index=i,
                text=doc.text[c0:c1], n_tokens=end - start, start_char=c0, end_char=c1,
                page_start=doc.page_at(c0), page_end=doc.page_at(max(c0, c1 - 1)),
            ))
            if end == n:
                break
            start += step
            i += 1
    return chunks
