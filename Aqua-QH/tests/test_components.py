import unittest
import numpy as np
from scipy.optimize import check_grad
from fish_qhprp.step1_hypergraph_build import build_hypergraph, HypergraphConfig, MockEncoder
from fish_qhprp.step2_hyper_prp import PairwiseSchedule
from fish_qhprp.step3_bt_aggregation import aggregate_soft_bt, loss_and_gradient
from fish_qhprp.rank_flan import MockJudge
from fish_qhprp.eval_results import ndcg_at_k
from fish_qhprp.utils import load_candidates


class ComponentsTest(unittest.TestCase):
    def setUp(self):
        self.encoder = MockEncoder()
        self.data = load_candidates('examples/synthetic_candidates.json', top_k=5)

    def test_hypergraph_overlap_and_fallback(self):
        q = self.data[0]
        graph = build_hypergraph(self.encoder.encode(q['texts']),
                                 self.encoder.encode([q['query']])[0],
                                 HypergraphConfig(num_topics=3, membership_threshold=0.01))
        self.assertEqual(graph.sparse.shape, (5, 3))
        self.assertTrue(np.all(graph.sparse.sum(axis=1) > 0))
        self.assertTrue(np.all((graph.sparse > 0).sum(axis=1) <= 3))
        np.testing.assert_allclose(graph.dense.sum(axis=1), 1)
        self.assertTrue(np.any((graph.sparse > 0).sum(axis=1) > 1))

    def test_no_overlap(self):
        q = self.data[0]
        graph = build_hypergraph(self.encoder.encode(q['texts']),
                                 self.encoder.encode([q['query']])[0],
                                 HypergraphConfig(num_topics=3), overlap=False)
        self.assertTrue(np.all((graph.sparse > 0).sum(axis=1) == 1))

    def test_coverage_and_connected(self):
        q = self.data[0]
        n = len(q['docids'])
        sch = PairwiseSchedule(q['query'], q['texts'], np.eye(n), q['bm25_scores'], MockJudge())
        records = sch.schedule([[i] for i in range(n)], min_degree=1)
        self.assertGreaterEqual(len(records), n-1)
        degrees = np.zeros(n, dtype=int)
        parent = list(range(n))
        def root(u):
            while parent[u] != u:
                u = parent[u]
            return u
        for rec in records:
            u, v = rec['u_idx'], rec['v_idx']
            degrees[u] += 1
            degrees[v] += 1
            parent[root(v)] = root(u)
        self.assertTrue(np.all(degrees >= 1))
        self.assertEqual(len({root(i) for i in range(n)}), 1)
        self.assertEqual(len(records), len({(r['u_idx'],r['v_idx']) for r in records}))

    def test_bt_gradient(self):
        records = [{'u_idx':0,'v_idx':1,'prob_u_win':0.8},
                   {'u_idx':1,'v_idx':2,'prob_u_win':0.45}]
        error = check_grad(lambda x: loss_and_gradient(x, records)[0],
                           lambda x: loss_and_gradient(x, records)[1],
                           np.array([0.1, -0.2, 0.3]))
        self.assertLess(error, 1e-6)
        theta = aggregate_soft_bt(3, records)
        self.assertAlmostEqual(theta.mean(), 0.0, places=9)
        self.assertGreater(theta[0], theta[1])

    def test_ndcg(self):
        qrels = {'a':2, 'b':1, 'c':0}
        self.assertAlmostEqual(ndcg_at_k(['a','b','c'], qrels), 1.0)
        self.assertLess(ndcg_at_k(['c','b','a'], qrels), 1.0)


if __name__ == '__main__':
    unittest.main()
