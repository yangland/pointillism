# attacks/base_attack.py
from typing import Optional
import torch
from torch.utils.data import Dataset

class BaseAttack:
    """
    Minimal interface for attack helpers.
    Methods:
      - set_global_model(model)           # server sets latest global model each round
      - client_search_trigger(loader)     # called by malicious client; returns local_trigger (tensor)
      - submit_client_trigger(trigger)    # client -> helper submit trigger for pooling
      - finalize_trigger(agg_mode="mean") # server-call to aggregate submitted triggers -> final trigger
      - prepare_evalset(test_set, per_class) -> Dataset
    """
    def __init__(self, cfg: dict):
        self.cfg = dict(cfg or {})
        self.device = None
        self._submitted_triggers = []

    def set_global_model(self, model: torch.nn.Module):
        self.global_model = model

    def client_search_trigger(self, loader) -> Optional[torch.Tensor]:
        raise NotImplementedError

    def submit_client_trigger(self, trig: torch.Tensor):
        # store on CPU to avoid device issues
        self._submitted_triggers.append(trig.detach().cpu())

    def finalize_trigger(self, agg_mode: str = "mean"):
        if not self._submitted_triggers:
            return None
        # default: element-wise mean
        stacked = torch.stack(self._submitted_triggers, dim=0)
        if agg_mode == "mean":
            self.trigger = stacked.mean(dim=0)
        elif agg_mode == "median":
            self.trigger = stacked.median(dim=0).values
        else:
            raise ValueError("unsupported agg_mode")
        # clear submitted list
        self._submitted_triggers = []
        return self.trigger

    def prepare_evalset(self, test_set: Dataset, per_class: int = 50):
        raise NotImplementedError
