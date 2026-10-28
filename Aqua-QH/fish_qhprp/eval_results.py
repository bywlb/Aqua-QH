
def read_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    """TREC qrels: qid 0 docid relevance, whitespace separated."""
    result: dict[str, dict[str, int]] = {}
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if len(fields) != 4:
            raise ValueError(f"{path}:{lineno}: expected qid 0 docid rel")
        qid, _, docid, rel = fields
        result.setdefault(qid, {})[docid] = int(rel)
    return result


def ndcg_at_k(ranked_docids: list[str], qrels: dict[str, int], k: int = 10) -> float:
    if not qrels:
        raise ValueError("No judgments for this query")
    gains = np.array([2**qrels.get(docid, 0)-1 for docid in ranked_docids[:k]], dtype=float)
    discounts = np.log2(np.arange(2, 2+len(gains)))
    dcg = float(np.sum(gains / discounts))
    ideal = np.sort([2**rel-1 for rel in qrels.values()])[::-1][:k]
    idcg = float(np.sum(np.asarray(ideal) / np.log2(np.arange(2, 2+len(ideal)))))
    return dcg/idcg if idcg > 0 else 0.0


def evaluate_ranking(ranking_json: str | Path, qrels_file: str | Path) -> dict:
    rows = load_json(ranking_json)
    qrels = read_qrels(qrels_file)
    scores = {}
    for row in rows:
        qid = str(row["qid"])
        if qid not in qrels:
            raise ValueError(f"Missing qrels for query {qid}; refuse partial average")
        scores[qid] = ndcg_at_k([str(d["docid"]) for d in row["ranking"]], qrels[qid])
    return {"queries": len(scores), "mean_ndcg_at_10": float(np.mean(list(scores.values()))),
            "per_query_ndcg_at_10": scores}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = evaluate_ranking(args.ranking, args.qrels)
    if args.output:
        save_json(result, args.output)
    print(f"Queries: {result['queries']}; NDCG@10: {result['mean_ndcg_at_10']:.4f}")


if __name__ == "__main__":
    main()
