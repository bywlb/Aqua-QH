
def rerank_queries(queries: list[dict], encoder, judge,
                   graph_config: HypergraphConfig = HypergraphConfig(),
                   ablation: str = "full", lambda_reg: float = 1e-4,
                   max_iter: int = 500, min_degree: int = 1) -> list[dict]:
    if ablation not in ABLATIONS:
        raise ValueError(f"Unknown ablation: {ablation}")
    rows = []
    for idx, item in enumerate(queries, start=1):
        t0 = time.monotonic()
        n = len(item["docids"])
        embeddings = encoder.encode(item["texts"])
        query_embedding = encoder.encode([item["query"]])[0]
        graph = build_hypergraph(
            embeddings, query_embedding, graph_config,
            query_attention=ablation != "no_query_attention",
            overlap=ablation != "no_overlap",
        )
        scheduler = PairwiseSchedule(
            item["query"], item["texts"], graph.sparse,
            item["bm25_scores"], judge,
            hard_labels=ablation == "no_probabilistic_preference",
        )
        records = scheduler.schedule(
            graph.hyperedges, local=ablation != "no_hyperedge_local", min_degree=min_degree,
        )
        theta = aggregate_soft_bt(n, records, lambda_reg=lambda_reg, max_iter=max_iter)
        order = sorted(range(n), key=lambda i: (-theta[i], i))
        rows.append({
            "qid": item["qid"], "query": item["query"],
            "ranking": [{"docid": item["docids"][i], "rank": rank,
                         "score": float(theta[i])} for rank, i in enumerate(order, 1)],
            "num_candidates": n,
            "num_hyperedges": sum(bool(edge) for edge in graph.hyperedges),
            "num_unique_pairs": scheduler.num_unique_pairs,
            "num_comparison_requests": scheduler.requests,
            "num_model_forward_calls": 2 * scheduler.num_unique_pairs,
            "seconds": round(time.monotonic() - t0, 3),
        })
        print(f"[{idx}/{len(queries)}] {item['qid']}: {n} docs, {len(records)} unique pairs")
    return rows


def save_run(rows: list[dict], output_dir: Path, name: str, metadata: dict) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking_json = output_dir / f"{name}.ranking.json"
    trec_file = output_dir / f"{name}.trec"
    save_json(rows, ranking_json)
    with trec_file.open("w", encoding="utf-8") as out:
        for row in rows:
            for hit in row["ranking"]:
                out.write(f"{row['qid']} Q0 {hit['docid']} {hit['rank']} {hit['score']:.9f} Aqua-QHPRP\n")
    save_json(metadata, output_dir / f"{name}.metadata.json")
    return ranking_json, trec_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aqua-QHPRP research-reference reranker")
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--input", help="Single BM25 JSON list of query candidates")
    sources.add_argument("--input-dir", help="Directory containing one BM25 JSON per subtask")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--qrels", help="Single qrels path (with --input)")
    parser.add_argument("--qrels-dir", help="Directory of qrels_<subtask>.txt (with --input-dir)")
    parser.add_argument("--judge", choices=["flan", "mock"], default="flan")
    parser.add_argument("--encoder", choices=["bge", "mock"], default="bge")
    parser.add_argument("--model", default="google/flan-t5-large")
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--prompt-file", help="UTF-8 prompt with {query}, {doc_a}, {doc_b}")
    parser.add_argument("--ablation", choices=ABLATIONS, default="full")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--num-topics", type=int, default=15)
    parser.add_argument("--tau-doc", type=float, default=0.8)
    parser.add_argument("--tau-query", type=float, default=0.5)
    parser.add_argument("--member-threshold", type=float, default=0.08)
    parser.add_argument("--max-memberships", type=int, default=3)
    parser.add_argument("--min-degree", type=int, default=1)
    parser.add_argument("--lambda-reg", type=float, default=1e-4)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def execute(args: argparse.Namespace) -> None:
    if args.judge == "mock" or args.encoder == "mock":
        print("WARNING: mock components are synthetic smoke-test substitutes; do not report these as paper results.")
    judge = MockJudge() if args.judge == "mock" else FlanT5Judge(
        model_name=args.model,
        prompt=Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else DEFAULT_PROMPT,
    )
    encoder = MockEncoder() if args.encoder == "mock" else BgeEncoder(args.embedding_model)
    np.random.seed(args.seed)
    config = HypergraphConfig(args.num_topics, args.tau_doc, args.tau_query,
                              args.member_threshold, args.max_memberships, args.seed)
    input_files = [Path(args.input)] if args.input else sorted(Path(args.input_dir).glob("*.json"))
    if not input_files:
        raise FileNotFoundError("No input JSON files found")
    task_scores = {}
    for path in input_files:
        name = path.stem
        queries = load_candidates(path, top_k=args.top_k)
        rows = rerank_queries(queries, encoder, judge, config, args.ablation,
                              args.lambda_reg, args.max_iter, args.min_degree)
        root = Path(args.output_dir) / args.ablation
        metadata = {
            "implementation": "research-reference; not verified as original experimental snapshot",
            "input": str(path), "ablation": args.ablation,
            "judge": args.judge, "encoder": args.encoder,
            "model": args.model if args.judge == "flan" else "MOCK",
            "embedding_model": args.embedding_model if args.encoder == "bge" else "MOCK",
            "num_topics": args.num_topics, "tau_doc": args.tau_doc,
            "tau_query": args.tau_query, "member_threshold": args.member_threshold,
            "max_memberships": args.max_memberships, "min_degree": args.min_degree,
            "lambda_reg": args.lambda_reg, "max_iter": args.max_iter,
            "seed": args.seed, "top_k": args.top_k,
            "prompt": str(args.prompt_file) if args.prompt_file else "code_default",
        }
        ranking_file, trec_file = save_run(rows, root, name, metadata)
        print(f"Saved: {ranking_file} and {trec_file}")
        qrels_path = Path(args.qrels) if args.qrels else (Path(args.qrels_dir) / f"qrels_{name}.txt" if args.qrels_dir else None)
        if qrels_path:
            result = evaluate_ranking(ranking_file, qrels_path)
            save_json(result, root / f"{name}.evaluation.json")
            task_scores[name] = result["mean_ndcg_at_10"]
            print(f"{name}: query-average NDCG@10={task_scores[name]:.4f}")
    if task_scores:
        # Never call a partial task set the official nine-task macro average.
        summary = {"subtasks": task_scores, "evaluated_subtask_count": len(task_scores)}
        if len(task_scores) == 9:
            summary["macro_averaged_ndcg_at_10"] = float(np.mean(list(task_scores.values())))
        save_json(summary, Path(args.output_dir) / args.ablation / "summary.json")


def main():
    execute(build_parser().parse_args())


if __name__ == "__main__":
    main()
