# Input and evaluation data

The reference implementation accepts the original BM25 retrieval JSON layout with one query and its ranked hits per entry. The top-level query ID may be the explicit `qid`, or a legacy `hits[0].qid` field. Each passage is keyed by `docid`, and a single query cannot repeat a document ID.

```json
[
  {
    "qid": "q0001",
    "query": "Example query text",
    "hits": [
      {"docid": "passage-001", "content": "Candidate passage", "score": 12.5},
      {"docid": "passage-002", "content": "Another passage", "score": 11.4}
    ]
  }
]
```

**Input is already retrieved**: the pipeline does **not** build the original corpus index or run BM25. Supply an identical Top-100 candidate pool for all compared reranking methods. The `score` and order are from the external BM25 retriever. No gold relevance labels are used during ranking.

For nine tasks, save each in `data/processed/<task_name>.json` using exactly these stems:

```
fact_disease
fact_pharmacology
fact_prevention
multi-hop_ecology
multi-hop_environment
multi-hop_pathology
reasoning_environment
reasoning_pharmacological
reasoning_symptoms
```

Gold qrels are standard four-column whitespace-separated lines `qid 0 docid rel` with `rel` in 0, 1, 2. Save to `data/qrels/qrels_<task_name>.txt`. All evaluated queries must have qrels; missing qrels cause an explicit error rather than a misleading partial score. Evaluations consider unlisted candidate docs nonrelevant; ideally judgments cover the complete Top-100 pool.

For each task the code writes:

- `<task>.ranking.json`: per-query ranked docs and pair scheduling statistics
- `<task>.trec`: TREC run file
- `<task>.metadata.json`: run parameters and provenance warning
- `<task>.evaluation.json`: if qrels were supplied

If all nine task qrels are supplied, `summary.json` includes the macro average of the nine task-level NDCG@10 values. If fewer are supplied, no official macro score is assigned.

## Rights and availability

The paper's corpus was compiled from papers and professional books. Extracted full text and chunked passages are not automatically redistributable. Release only data for which you have permission, and link to original materials where appropriate. The sample data supplied here are deliberately synthetic and should not be mistaken for published benchmark examples.
