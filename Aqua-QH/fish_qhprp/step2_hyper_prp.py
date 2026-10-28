


class PreferenceJudge(Protocol):
    def compare(self, query: str, doc_a: str, doc_b: str) -> float: ...


class PairwiseSchedule:
    def __init__(self, query: str, texts: list[str], memberships: np.ndarray,
                 bm25_scores: list[float], judge: PreferenceJudge, hard_labels: bool = False):
        self.query, self.texts, self.judge = query, texts, judge
        self.H = np.asarray(memberships, dtype=float)
        self.n = len(texts)
        self.bm25 = np.asarray(bm25_scores, dtype=float)
        self.hard_labels = hard_labels
        self._pairs: dict[tuple[int, int], float] = {}
        self.requests = 0

    def preference(self, u: int, v: int) -> float:
        if u == v:
            return 0.5
        self.requests += 1
        lo, hi = sorted((u, v))
        key = (lo, hi)
        if key not in self._pairs:
            p = float(self.judge.compare(self.query, self.texts[lo], self.texts[hi]))
            if not np.isfinite(p) or not 0 <= p <= 1:
                raise ValueError("Judge.compare must return a finite probability in [0,1]")
            self._pairs[key] = float(p >= 0.5) if self.hard_labels else p
        return self._pairs[key] if u == lo else 1 - self._pairs[key]

    def _is_better(self, u: int, v: int) -> bool:
        p = self.preference(u, v)
        return p > 0.5 or (p == 0.5 and u < v)

    def heapsort(self, indices: list[int]) -> list[int]:
        """Ascending min-heap extraction yields most relevant first."""
        arr = indices.copy()

        def heapify(n: int, i: int) -> None:
            while True:
                weakest = i
                left, right = 2 * i + 1, 2 * i + 2
                if left < n and self._is_better(arr[weakest], arr[left]):
                    weakest = left
                if right < n and self._is_better(arr[weakest], arr[right]):
                    weakest = right
                if weakest == i:
                    return
                arr[i], arr[weakest] = arr[weakest], arr[i]
                i = weakest

        for i in range(len(arr) // 2 - 1, -1, -1):
            heapify(len(arr), i)
        for i in range(len(arr) - 1, 0, -1):
            arr[0], arr[i] = arr[i], arr[0]
            heapify(i, 0)
        return arr

    def schedule(self, hyperedges: list[list[int]], local: bool = True,
                 min_degree: int = 1) -> list[dict]:
        if min_degree < 0:
            raise ValueError("min_degree must be nonnegative")
        if local:
            # Same comparison may occur in several hyperedges; cached by unordered pair.
            for members in hyperedges:
                if len(members) >= 2:
                    order = sorted(members, key=lambda i: (-self.bm25[i], i))
                    self.heapsort(order)
        else:
            order = sorted(range(self.n), key=lambda i: (-self.bm25[i], i))
            self.heapsort(order)
        self._repair_coverage(min_degree)
        self._repair_connectivity()
        return self.records()

    def _degree(self) -> list[int]:
        degree = [0] * self.n
        for u, v in self._pairs:
            degree[u] += 1
            degree[v] += 1
        return degree

    def _candidate_order(self, u: int) -> list[int]:
        """Prefer shared high-weight hyperedges and similar initial ranks."""
        scores = np.asarray(self.H, dtype=float)
        shared = np.sum(np.minimum(scores[u], scores), axis=1)
        return sorted((v for v in range(self.n) if v != u),
                      key=lambda v: (-shared[v], abs(u-v), v))

    def _repair_coverage(self, min_degree: int) -> None:
        if self.n < 2:
            return
        target = min(min_degree, self.n - 1)
        for u in range(self.n):
            while self._degree()[u] < target:
                options = [v for v in self._candidate_order(u)
                           if tuple(sorted((u, v))) not in self._pairs]
                if not options:
                    break
                self.preference(u, options[0])

    def _repair_connectivity(self) -> None:
        if self.n <= 1:
            return
        parent = list(range(self.n))

        def root(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u

        def merge(u, v):
            ru, rv = root(u), root(v)
            if ru != rv:
                parent[rv] = ru

        for u, v in self._pairs:
            merge(u, v)
        while len({root(i) for i in range(self.n)}) > 1:
            # Strongest soft-membership overlap across components; deterministic tie-break.
            options = ((float(np.minimum(self.H[u], self.H[v]).sum()),
                        -abs(u-v), -u, -v, u, v)
                       for u in range(self.n) for v in range(u+1, self.n)
                       if root(u) != root(v))
            _, _, _, _, u, v = max(options)
            self.preference(u, v)
            merge(u, v)

    def records(self) -> list[dict]:
        return [{"u_idx": u, "v_idx": v, "prob_u_win": p}
                for (u, v), p in sorted(self._pairs.items())]

    @property
    def num_unique_pairs(self) -> int:
        return len(self._pairs)
