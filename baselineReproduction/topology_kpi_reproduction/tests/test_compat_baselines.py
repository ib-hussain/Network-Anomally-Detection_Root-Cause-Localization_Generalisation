import unittest
import numpy as np
import torch

from topokpi.deep_baselines import build_deep_baseline
from topokpi.rca_baselines import rank_rca, NezhaRestrictedCompat
from topokpi.explainability import explainability_metrics
from topokpi.testbed import default_schedule, event_labels, topology_edges


class DeepBaselineTests(unittest.TestCase):
    def test_all_detection_baselines_are_finite(self):
        x=torch.randn(2,20,8); y=torch.randn(2,8)
        names=["OmniAnomaly","MSCRED","USAD","THOC","MTAD-GAT","GDN","InterFusion","Anomaly Transformer","TranAD","DCdetector"]
        for name in names:
            with self.subTest(name=name):
                model=build_deep_baseline(name,8,20); loss=model.loss(x,y); score=model.score(x,y)
                self.assertTrue(torch.isfinite(loss)); self.assertEqual(tuple(score.shape),(2,)); self.assertTrue(torch.isfinite(score).all())


class RCABaselineTests(unittest.TestCase):
    def test_all_non_neural_rca_rankers(self):
        rng=np.random.default_rng(2); series=np.abs(rng.normal(size=(20,6))); metrics=np.abs(rng.normal(size=(20,6,8)))
        topology=np.zeros((6,6)); topology[np.arange(5),np.arange(1,6)]=1
        for name in ["OLS-residual","VAR(1)-residual","CauseInfer","MonitorRank","CloudRanger","MicroCause","TraceRCA (restricted)"]:
            with self.subTest(name=name):
                ranking=rank_rca(name,series,topology,metrics)
                self.assertEqual(sorted(ranking),list(range(6)))

    def test_nezha_restricted_shape(self):
        model=NezhaRestrictedCompat(8,hidden=16,heads=4,layers=1); logits=model(torch.randn(2,10,6,8),np.eye(6)); self.assertEqual(tuple(logits.shape),(2,6))

    def test_explanation_metrics(self):
        contributions=np.zeros((1,3,4)); contributions[0,1]=[.1,.7,.1,.1]
        result=explainability_metrics(contributions,np.array([1]),[[1]],[[(1,2)]],[[(1,2),(2,0)]])
        self.assertEqual(result["kpi_hit@1"],1); self.assertEqual(result["path_any_overlap"],1)

    def test_campaign_specification(self):
        edges=topology_edges(12); self.assertGreaterEqual(len(edges),11)
        events=default_schedule(12,300,event_duration=10); labels,roots,faults,ids,w=event_labels(12,300,events)
        self.assertEqual(len(events),6); self.assertEqual(w.shape,(300,12,12)); self.assertGreater(labels.sum(),0)


if __name__ == "__main__": unittest.main()
