
def save_json(obj: Any, path: str | Path) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(values)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)


def load_candidates(path: str | Path, top_k: int = 100) -> list[dict]:
    """Parse the original BM25-list structure or an equivalent explicit-qid structure.

    Each query: {"qid": "...", "query": "...", "hits": [{"docid": ...,
    "content": ..., "score": ...}, ...]}. A legacy first-hit ``qid`` is accepted.
    Document IDs must be unique *within* each query. Queries must have stable IDs.
    """
    raw = load_json(path)
    if isinstance(raw, dict) and "queries" in raw:
        raw = raw["queries"]
    if not isinstance(raw, list):
        raise ValueError("Candidate file must contain a list of query objects")
    output, qids = [], set()
    for idx, item in enumerate(raw):
        hits = item.get("hits", [])[:top_k]
        if not hits:
            continue
        qid = str(item.get("qid") or hits[0].get("qid") or "")
        if not qid:
            raise ValueError(f"Query at index {idx} has no qid")
        if qid in qids:
            raise ValueError(f"Duplicate qid: {qid}")
        qids.add(qid)
        docids = [str(h["docid"]) for h in hits]
        if len(set(docids)) != len(docids):
            raise ValueError(f"Query {qid} has duplicate docids")
        if not str(item.get("query", "")).strip():
            raise ValueError(f"Query {qid} is empty")
        output.append({
            "qid": qid, "query": str(item["query"]),
            "docids": docids,
            "texts": [str(h.get("content", "")) for h in hits],
            "bm25_scores": [float(h.get("score", 0.0)) for h in hits]
        })
    if not output:
        raise ValueError("No usable queries with hits")
    return output


def cosine_to_centers(vectors: np.ndarray, centers: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    a = vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12)
    b = centers / np.maximum(np.linalg.norm(centers, axis=-1, keepdims=True), 1e-12)
    return a @ b.T
