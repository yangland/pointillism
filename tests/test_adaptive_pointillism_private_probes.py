import inspect
import unittest

from aggregation import pointillism_fl
from fl.client import FLClient
from fl.server import FLServer


class AdaptivePointillismPrivateProbeTest(unittest.TestCase):
    def test_audit_uses_fresh_private_state(self):
        source = inspect.getsource(pointillism_fl.audit_client_states)

        self.assertIn("private_probe_seed", source)
        self.assertIn("PointillismFLState()", source)
        self.assertIn('audit_cfg["seed"] = int(private_probe_seed)', source)
        self.assertNotIn("_get_state", source)
        self.assertNotIn("out_dir", source)

    def test_batched_audit_uses_one_private_probe_pool(self):
        source = inspect.getsource(
            pointillism_fl.audit_adaptive_candidate_states
        )

        self.assertIn("private_probe_seed", source)
        self.assertIn("PointillismFLState()", source)
        self.assertEqual(source.count("build_round_probe_pool("), 1)
        self.assertEqual(source.count("score_probe_specs("), 2)

    def test_server_invokes_one_batched_audit(self):
        source = inspect.getsource(
            FLServer._select_adaptive_pointillism_candidates
        )

        self.assertIn("probe_seed_offset", source)
        self.assertEqual(source.count("audit_adaptive_candidate_states("), 1)
        self.assertEqual(source.count("private_probe_seed=private_probe_seed"), 1)
        self.assertNotIn("audit_client_states(", source)
        self.assertNotIn("out_dir=self.out_dir", source)

    def test_clients_train_two_endpoints_and_interpolate_candidates(self):
        source = inspect.getsource(
            FLClient.prepare_adaptive_pointillism_candidates
        )

        self.assertIn("for ratio in [None, endpoint_poison_frac]", source)
        self.assertIn("torch.lerp", source)
        self.assertIn('adaptive.get("endpoint_poison_frac", 0.5)', source)

