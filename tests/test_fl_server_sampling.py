import numpy as np

from fl.server import FLServer


def test_fixed_rate_sampling_keeps_three_malicious_and_seven_benign():
    server = FLServer.__new__(FLServer)
    server.cfg = {
        "clients_setting": {
            "clients_per_round": 10,
            "mali_rate": 0.3,
            "sampling_method": "fixed_rate",
        }
    }
    server.rng = np.random.default_rng(2026)
    server.malicious_ids = list(range(30))
    server.benign_ids = list(range(30, 100))

    for _ in range(100):
        selected = server._sample_clients_once(num_clients=100)
        assert len(selected) == 10
        assert len(set(selected)) == 10
        assert sum(cid < 30 for cid in selected) == 3
        assert sum(cid >= 30 for cid in selected) == 7
