
def build_hypergraph(doc_vectors: np.ndarray, query_vector: np.ndarray,
                     config: HypergraphConfig = HypergraphConfig(),
                     query_attention: bool = True, overlap: bool = True) -> Hypergraph:
    doc_vectors = np.asarray(doc_vectors, dtype=float)
    query_vector = np.asarray(query_vector, dtype=float).reshape(-1)
    if doc_vectors.ndim != 2 or doc_vectors.shape[0] < 1 or doc_vectors.shape[1] != query_vector.shape[0]:
        raise ValueError("Embeddings must have shapes (num_docs, dim) and (dim,)")
    if config.tau_doc <= 0 or config.tau_query <= 0:
        raise ValueError("Softmax temperatures must be positive")
    if config.max_memberships < 1 or config.num_topics < 1:
        raise ValueError("num_topics and max_memberships must be positive")
    k = min(config.num_topics, len(doc_vectors))
    centers = KMeans(n_clusters=k, n_init=10, random_state=config.seed).fit(doc_vectors).cluster_centers_ if k > 1 else np.mean(doc_vectors, axis=0, keepdims=True)
    base = softmax(cosine_to_centers(doc_vectors, centers) / config.tau_doc, axis=1)
    query_weights = softmax(cosine_to_centers(query_vector.reshape(1, -1), centers)[0] / config.tau_query)
    weighted = base * query_weights[None, :] if query_attention else base
    dense = weighted / np.sum(weighted, axis=1, keepdims=True)
    sparse = np.zeros_like(dense)
    for i, weights in enumerate(dense):
        ranked = sorted(range(k), key=lambda c: (-weights[c], c))
        if overlap:
            selected = [c for c in ranked if weights[c] >= config.membership_threshold][:config.max_memberships]
        else:
            selected = ranked[:1]
        if not selected:
            selected = ranked[:1]
        sparse[i, selected] = weights[selected]
    hyperedges = [[i for i in range(len(doc_vectors)) if sparse[i, c] > 0] for c in range(k)]
    return Hypergraph(dense, sparse, hyperedges, query_weights)


class BgeEncoder:
    """Lazy external embedding dependency: no model download for mock tests."""
    def __init__(self, model: str = "BAAI/bge-m3"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install sentence-transformers for real embeddings") from exc
        self.encoder = SentenceTransformer(model)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False))


class MockEncoder:
    """Deterministic toy embedding for offline smoke tests only; NOT experimental."""
    def __init__(self, dim: int = 64):
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        import hashlib
        vectors = []
        for text in texts:
            vec = np.zeros(self.dim)
            for i in range(len(text)):
                gram = text[i:i+2]
                slot = int.from_bytes(hashlib.sha256(gram.encode()).digest()[:4], 'little') % self.dim
                vec[slot] += 1.0
            norm = np.linalg.norm(vec)
            vectors.append(vec / norm if norm else vec)
        return np.asarray(vectors)
