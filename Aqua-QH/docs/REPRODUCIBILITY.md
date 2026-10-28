# Reproducibility and release notes

## What is implemented and tested

- Query-aware soft memberships from cosine similarity of BGE-M3 embeddings and K-Means centers (document temperature 0.8, query temperature 0.5).
- Query-conditioned normalization, selection threshold 0.08, at most 3 hyperedge memberships, fallback to at least one membership.
- Independent hyperedge-local HeapSort schedules, global cache of unordered LLM comparisons, minimum comparison degree 1 and graph connectivity repair.
- Forward/reverse FLAN-T5 A/B token-based preference probabilities.
- Soft-label BT cross entropy with ordinary L2 penalty (lambda=1e-4), SciPy L-BFGS-B (maxiter=500) and mean-centered theta.
- Four named paper-style ablations, TREC export, NDCG@10 and nine-task macro averaging.

Algorithmic and gradient tests are included. The offline smoke test uses model stand-ins; it does not validate FLAN-T5 output quality, any performance gain, or the reported paper values.

## What is *not* included / verified

1. **No original AquaQA-Rerank real inputs / qrels / corpus** were provided for this release. No copyright-sensitive passages have been published.
2. **No verified final experimental scripts** were provided. Only older versions of five Python modules and some earlier ablation runners were available. The older versions disagree with the current manuscript on the lexical hyperedge, scheduling, BT penalty, iteration count, and ablation naming.
3. The FLAN-T5 prompt and passage truncation configuration used to obtain the final paper's table are **not confirmed**. The generic prompt in `rank_flan.py` is an implementation default, not asserted to be the original prompt. Use `--prompt-file` and confirmed limits if exact reproduction is required.
4. **Published Table 5 / Table 6 scores have not been rerun from this release.** There are no fabricated `results.csv` or purported trained weights. Reported numbers belong to the manuscript, not to the included synthetic example.
5. Baseline implementations (PRP-AllPair, PRP-HeapSort, PRP-Graph) and the upstream BM25 index are not included.
6. Reproducibility across FLAN-T5-Large/XL/XXL is subject to memory, model version, library versions, prompt and random seed. The manuscript's claimed hardware/software environment has not been cross-checked with run logs.

## Before attributing the manuscript's experimental results to this repository

- Verify that the publication describes the algorithm contained here, or replace this reference implementation with the **exact source used to run the paper**.
- Verify dataset rights, provide legally distributable inputs or access instructions and actual qrels.
- Record the original FLAN-T5 prompt, model weight revision, tokenizer settings, text truncation, random seeds and dependencies.
- Rerun all 9 tasks under all 3 models and compare results to manuscript Table 5 and Table 6.
- Confirm authorship, code ownership and license terms; replace generic maintainer metadata.

**Publishing this reference repository as a documented implementation is reasonable; claiming the paper's numbers are reproduced by it is not yet supported.**
