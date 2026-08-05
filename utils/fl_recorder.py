# utils/fl_recorder.py
from __future__ import annotations

import os
import csv
from typing import Any, Dict, List, Optional


class Recorder:
    """
    Minimal FL recorder (no RandAudit).

    Files created in out_dir:
      - run.log
      - server_metrics.csv
      - selections.csv
      - selection_decisions.csv
      - selection_metrics.csv
      - round_times.csv
      - adaptive_pointillism_search.csv
      - adaptive_pointillism_summary.csv
      - pointillism_clients.csv
      - pointillism_alignment.csv
      - pointillism_bank.csv
    """

    def __init__(self, out_dir: str, save_every: int = 1, log_filename: str = "run.log"):
        self.out_dir = str(out_dir)
        os.makedirs(self.out_dir, exist_ok=True)

        self.save_every = int(save_every)

        self.log_path = os.path.join(self.out_dir, log_filename)
        self.metrics_path = os.path.join(self.out_dir, "server_metrics.csv")
        self.sel_path = os.path.join(self.out_dir, "selections.csv")
        self.selection_decisions_path = os.path.join(
            self.out_dir, "selection_decisions.csv"
        )
        self.selection_metrics_path = os.path.join(
            self.out_dir, "selection_metrics.csv"
        )
        self.round_times_path = os.path.join(self.out_dir, "round_times.csv")
        self.adaptive_pt_path = os.path.join(self.out_dir, "adaptive_pointillism_search.csv")
        self.adaptive_pt_summary_path = os.path.join(self.out_dir, "adaptive_pointillism_summary.csv")

        # Pointillism CSVs
        self.pt_clients_path = os.path.join(self.out_dir, "pointillism_clients.csv")
        self.pt_align_path = os.path.join(self.out_dir, "pointillism_alignment.csv")
        self.pt_bank_path = os.path.join(self.out_dir, "pointillism_bank.csv")

        # init empty log
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("")

        # init CSV headers
        self._init_csv(self.metrics_path, ["round", "acc", "asr", "mali_weight"])
        self._init_csv(self.sel_path, ["round", "cid", "is_malicious"])
        self._init_csv(
            self.selection_decisions_path,
            ["round", "cid", "is_malicious", "is_selected", "agg_weight"],
        )
        self._init_csv(
            self.selection_metrics_path,
            [
                "round", "tp", "fn", "fp", "tn", "num_malicious",
                "num_benign", "fnr", "fpr",
            ],
        )
        self._init_csv(
            self.round_times_path,
            ["round", "defense", "train_s", "defense_s", "evaluation_s", "hook_s", "total_s"],
        )
        self._init_csv(self.adaptive_pt_path, [
            "round", "cid", "poison_ratio", "local_acc", "local_asr",
            "consensus_score", "tau_score", "score_margin", "pass_score",
            "kept_in_simulation", "local_loss",
        ])
        self._init_csv(self.adaptive_pt_summary_path, [
            "round", "cid", "submitted_ratio", "submitted_clean", "selected_acc",
            "selected_asr", "selected_score", "tau_score", "score_margin",
        ])

        self._init_csv(
            self.pt_clients_path,
            ["round", "cid", "keep", "cluster", "A_mean", "A_min", "A_max"],
        )
        self._init_csv(
            self.pt_align_path,
            ["round", "cid", "class_idx", "A_ij"],
        )
        self._init_csv(
            self.pt_bank_path,
            ["round", "class_idx", "rank", "probe_id", "w", "pct", "inv", "ms"],
        )

        # buffers (optional)
        self._metrics_buf: List[List[Any]] = []
        self._sel_buf: List[List[Any]] = []
        self._selection_decisions_buf: List[List[Any]] = []
        self._selection_metrics_buf: List[List[Any]] = []
        self._round_times_buf: List[List[Any]] = []
        self._pt_clients_buf: List[List[Any]] = []
        self._pt_align_buf: List[List[Any]] = []
        self._pt_bank_buf: List[List[Any]] = []

    # -------------------------
    # low-level helpers
    # -------------------------
    def _init_csv(self, path: str, header: List[str]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

    def _append_rows(self, path: str, rows: List[List[Any]]) -> None:
        if not rows:
            return
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerows(rows)

    # -------------------------
    # text logging
    # -------------------------
    def maybe_print(self, msg: str) -> None:
        # keeps your current server_min logging style
        print(msg)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(str(msg) + "\n")

    # -------------------------
    # existing server logs
    # -------------------------
    def log_metrics(
        self,
        rnd: int,
        acc: float,
        asr: Optional[float],
        mali_weight: Optional[float],
    ) -> None:
        row = [
            int(rnd),
            float(acc),
            "" if asr is None else float(asr),
            "" if mali_weight is None else float(mali_weight),
        ]
        self._metrics_buf.append(row)
        if int(rnd) % self.save_every == 0:
            self.flush()

    def log_selection(self, rnd: int, selected: List[Any], mal_flags: List[bool]) -> None:
        if len(selected) != len(mal_flags):
            raise ValueError("log_selection: selected and mal_flags length mismatch")

        for cid, is_mal in zip(selected, mal_flags):
            row = [int(rnd), cid, int(bool(is_mal))]
            self._sel_buf.append(row)

    def log_selection_outcomes(
        self,
        rnd: int,
        client_ids: List[Any],
        mal_flags: List[bool],
        agg_weights: List[float],
    ) -> None:
        """
        Log defense selection and its malicious-client confusion matrix.

        An excluded client (aggregation weight <= 0) is a positive malicious
        prediction. Therefore:
          FNR = malicious clients selected / malicious clients participating
          FPR = benign clients excluded / benign clients participating
        """
        if not (len(client_ids) == len(mal_flags) == len(agg_weights)):
            raise ValueError(
                "log_selection_outcomes: client_ids, mal_flags, and "
                "agg_weights length mismatch"
            )

        tp = fn = fp = tn = 0
        for cid, is_mal, weight in zip(client_ids, mal_flags, agg_weights):
            selected = float(weight) > 0.0
            malicious = bool(is_mal)
            self._selection_decisions_buf.append(
                [
                    int(rnd),
                    cid,
                    int(malicious),
                    int(selected),
                    float(weight),
                ]
            )
            if malicious and selected:
                fn += 1
            elif malicious:
                tp += 1
            elif selected:
                tn += 1
            else:
                fp += 1

        num_malicious = tp + fn
        num_benign = fp + tn
        fnr = float(fn) / float(num_malicious) if num_malicious else ""
        fpr = float(fp) / float(num_benign) if num_benign else ""
        self._selection_metrics_buf.append(
            [
                int(rnd), tp, fn, fp, tn, num_malicious, num_benign, fnr, fpr
            ]
        )

    def log_round_time(
        self,
        rnd: int,
        defense: str,
        train_s: float,
        defense_s: float,
        evaluation_s: float,
        hook_s: float,
        total_s: float,
    ) -> None:
        """Record synchronized end-to-end and phase wall-clock times."""
        self._round_times_buf.append(
            [
                int(rnd),
                str(defense),
                float(train_s),
                float(defense_s),
                float(evaluation_s),
                float(hook_s),
                float(total_s),
            ]
        )

    # -------------------------
    # pointillism logs
    # -------------------------
    def log_pointillism_clients(
        self,
        rnd: int,
        rows: List[Dict[str, Any]],
    ) -> None:
        """
        rows: list of dicts with keys:
          cid, keep (0/1), cluster (-1/0/1), A_mean, A_min, A_max
        """
        out: List[List[Any]] = []
        for r in rows:
            out.append(
                [
                    int(rnd),
                    r.get("cid"),
                    int(r.get("keep", 0)),
                    int(r.get("cluster", -1)),
                    float(r.get("A_mean", 0.0)),
                    float(r.get("A_min", 0.0)),
                    float(r.get("A_max", 0.0)),
                ]
            )
        self._pt_clients_buf.extend(out)

    def log_pointillism_alignment(
        self,
        rnd: int,
        cid_to_A: Dict[Any, List[float]],
    ) -> None:
        """
        cid_to_A: {cid: [A_0, A_1, ... A_{C-1}]}
        """
        out: List[List[Any]] = []
        for cid, Avec in cid_to_A.items():
            for j, aij in enumerate(Avec):
                out.append([int(rnd), cid, int(j), float(aij)])
        self._pt_align_buf.extend(out)

    def log_pointillism_bank(
        self,
        rnd: int,
        banks: Dict[int, List[Dict[str, Any]]],
    ) -> None:
        """
        banks: {class_idx: [BankEntry...]}
        BankEntry keys: probe_id, w, pct, inv, ms
        """
        out: List[List[Any]] = []
        for j, entries in banks.items():
            for rank, be in enumerate(entries):
                out.append(
                    [
                        int(rnd),
                        int(j),
                        int(rank),
                        int(be.get("probe_id")),
                        int(be.get("w")),
                        float(be.get("pct")),
                        int(1 if be.get("inv") else 0),
                        float(be.get("ms")),
                    ]
                )
        self._pt_bank_buf.extend(out)

    def log_adaptive_pointillism(self, rows: List[List[Any]]) -> None:
        self._append_rows(self.adaptive_pt_path, rows)

    def log_adaptive_pointillism_summary(self, rows: List[Dict[str, Any]]) -> None:
        keys = ["round", "cid", "submitted_ratio", "submitted_clean", "selected_acc",
                "selected_asr", "selected_score", "tau_score", "score_margin"]
        self._append_rows(self.adaptive_pt_summary_path, [[row.get(k, "") for k in keys] for row in rows])

    # -------------------------
    # flush
    # -------------------------
    def flush(self) -> None:
        # write and clear buffers
        self._append_rows(self.metrics_path, self._metrics_buf)
        self._append_rows(self.sel_path, self._sel_buf)
        self._append_rows(
            self.selection_decisions_path, self._selection_decisions_buf
        )
        self._append_rows(
            self.selection_metrics_path, self._selection_metrics_buf
        )
        self._append_rows(self.round_times_path, self._round_times_buf)
        self._append_rows(self.pt_clients_path, self._pt_clients_buf)
        self._append_rows(self.pt_align_path, self._pt_align_buf)
        self._append_rows(self.pt_bank_path, self._pt_bank_buf)

        self._metrics_buf.clear()
        self._sel_buf.clear()
        self._selection_decisions_buf.clear()
        self._selection_metrics_buf.clear()
        self._round_times_buf.clear()
        self._pt_clients_buf.clear()
        self._pt_align_buf.clear()
        self._pt_bank_buf.clear()
