import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch not installed")
class ModelTests(unittest.TestCase):
    def test_forward_and_loss(self):
        from topokpi.model import TopologyAwareSSL, ssl_loss
        model = TopologyAwareSSL(kpis=8, hidden=16, temporal_layers=1, graph_layers=1, heads=4, feedforward=32)
        x = torch.randn(2, 10, 6, 8)
        y = torch.randn(2, 6, 8)
        a = torch.eye(6)
        mask = torch.rand_like(x) < .15
        rec, pred, latent = model(x.masked_fill(mask, 0), a)
        self.assertEqual(tuple(rec.shape), (2, 10, 6, 8))
        self.assertEqual(tuple(pred.shape), (2, 6, 8))
        self.assertEqual(tuple(latent.shape), (2, 10, 6, 16))
        loss, parts = ssl_loss(model, x, x.masked_fill(mask, 0), mask, y, a)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIn("topology", parts)


if __name__ == "__main__":
    unittest.main()

