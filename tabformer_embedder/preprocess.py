"""Column encoders: categorical (per-value ids) and numeric (quantile bins).

Fits on a pandas DataFrame (optionally accumulated chunk by chunk for CSV
files) and transforms any DataFrame into a dense int32 matrix of global
vocabulary ids, one row per record, one column per *output field*.

nuFormer-style options (both enabled by default for arch="nuformer"):
  * ``decompose_dates``: datetime columns expand into dedicated small
    vocabularies -- month (12), day (31), weekday (7), hour (24) -- plus a
    quantile-binned year, instead of one bin over raw seconds.
  * ``sign_split``: numeric columns that contain negative values split into
    a sign token (inflow/outflow) + magnitude quantile bin (nuFormer's
    amount encoding).

Memory notes:
  * categorical vocabularies are plain Python dicts of canonical string keys
    (~ vocab_size entries, not nrows);
  * numeric buffers are float32 and are freed right after `finalize`;
  * the output matrix is int32 (4 bytes / token) instead of Python lists.
"""

import numpy as np
import pandas as pd

from .vocab import FieldVocab, UNK_ID, canon

__all__ = ["TabularPreprocessor"]


def _to_float(series):
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float32)


def _int_keys(vals):
    """Canonical keys for an integer-valued series (with NaN -> __nan__)."""
    s = pd.Series(vals).astype("Float64").astype("string")
    return s.fillna("__nan__").replace({"<NA>": "__nan__"}).to_numpy(dtype=object)


class TabularPreprocessor:
    """Fit/transform tabular data into global vocabulary ids."""

    DT_SUBS = ["month", "day", "weekday", "hour"]   # categorical parts
    DT_NUM = "year"                                  # quantile-binned part

    def __init__(self, num_bins=10, min_freq=1, datetime_cols=None,
                 decompose_dates=False, sign_split=False,
                 categorical_cols=None):
        self.num_bins = int(num_bins)
        self.min_freq = int(min_freq)
        self.datetime_cols = list(datetime_cols) if datetime_cols else []
        self.decompose_dates = bool(decompose_dates)
        self.sign_split = bool(sign_split)
        self.categorical_cols = list(categorical_cols) if categorical_cols else []

        self.columns = []          # original input columns, in order
        self.kinds = {}            # original col -> "cat" | "num" | "dt"
        self.out_columns = []      # expanded output fields, in order
        self.cat_maps = {}         # out col -> {canon_key: local_id}
        self.num_edges = {}        # out col -> np.ndarray of bin edges
        self.n_local = {}          # out col -> number of local ids
        self._signed = {}          # original num col -> bool (has negatives)
        self.vocab = FieldVocab()

        self._acc_cat = {}         # out col -> {key: count}   (during accumulate)
        self._acc_num = {}         # out col -> [np.float32]   (during accumulate)
        self._neg = {}             # original num col -> count of negatives

    # ------------------------------------------------------------------ fit
    def _is_dt(self, series, col):
        return (col in self.datetime_cols
                or pd.api.types.is_datetime64_any_dtype(series))

    def _kind(self, series, col=None):
        if col is not None and col in self.categorical_cols:
            return "cat"  # e.g. nominal code columns (MCC, Zip, ...)
        if col is not None and self._is_dt(series, col) and self.decompose_dates:
            return "dt"
        if pd.api.types.is_datetime64_any_dtype(series):
            return "num"
        if pd.api.types.is_numeric_dtype(series):
            return "num"
        # object columns that are fully numeric strings are still categorical:
        # keeping them categorical is safer (no coercion surprises)
        return "cat"

    def _as_float(self, series, col=None):
        if col is not None and self._is_dt(series, col):
            series = pd.to_datetime(series, errors="coerce")
        if pd.api.types.is_datetime64_any_dtype(series):
            s = series.to_numpy(dtype="datetime64[ns]").astype("int64")
            s = s.astype(np.float64)
            s[s < -6.2e17] = np.nan  # NaT sentinel window
            return s.astype(np.float32)
        return _to_float(series)

    def _as_dt(self, series, col):
        series = pd.Series(series).reset_index(drop=True)
        if not pd.api.types.is_datetime64_any_dtype(series):
            series = pd.to_datetime(series, errors="coerce")
        return series

    def accumulate(self, df):
        """First pass: collect value counts (cat) / raw values (num)."""
        if not self.columns:
            self.columns = list(df.columns)
            for col in self.columns:
                self.kinds[col] = self._kind(df[col], col)

        for col in self.columns:
            kind = self.kinds[col]
            if kind == "cat":
                acc = self._acc_cat.setdefault(col, {})
                for val, cnt in df[col].value_counts(dropna=False).items():
                    key = canon(val)
                    acc[key] = acc.get(key, 0) + int(cnt)
            elif kind == "num":
                vals = self._as_float(df[col], col)
                if self.sign_split:
                    self._neg[col] = self._neg.get(col, 0) + int((vals < 0).sum())
                    vals = np.abs(vals)
                self._acc_num.setdefault(col, []).append(vals)
            else:  # dt -> dedicated sub-vocabularies (nuFormer-style)
                d = self._as_dt(df[col], col)
                for part in self.DT_SUBS:
                    key = f"{col}#{part}"
                    acc = self._acc_cat.setdefault(key, {})
                    vals = getattr(d.dt, "dayofweek" if part == "weekday" else part)
                    for k, cnt in pd.Series(_int_keys(vals)).value_counts(dropna=False).items():
                        acc[k] = acc.get(k, 0) + int(cnt)
                self._acc_num.setdefault(f"{col}#{self.DT_NUM}", []).append(
                    d.dt.year.to_numpy(dtype=np.float32))

    def finalize(self, user_map=None):
        """Finish fitting: compute bin edges, local ids and global bases."""
        self.out_columns = []
        for col in self.columns:
            kind = self.kinds[col]
            if kind == "cat":
                self.out_columns.append(col)
            elif kind == "num":
                signed = self.sign_split and self._neg.get(col, 0) > 0
                self._signed[col] = bool(signed)
                if signed:
                    self.out_columns += [f"{col}#sign", f"{col}#bin"]
                else:
                    self.out_columns.append(col)
            else:  # dt
                self.out_columns += [f"{col}#{p}" for p in self.DT_SUBS]
                self.out_columns.append(f"{col}#{self.DT_NUM}")

        for out in self.out_columns:
            if out in self._acc_cat:
                acc = self._acc_cat[out]
                keys = [k for k, c in acc.items() if c >= self.min_freq]
                keys.sort()  # deterministic order
                self.cat_maps[out] = {k: i for i, k in enumerate(keys)}
                self.n_local[out] = len(keys)
                self._acc_cat[out] = None
            else:
                if out.endswith("#sign"):
                    self.cat_maps[out] = {"neg": 0, "pos": 1}
                    self.n_local[out] = 2
                    continue
                buf = (np.concatenate(self._acc_num[out])
                       if self._acc_num.get(out) else np.zeros(0, dtype=np.float32))
                finite = buf[np.isfinite(buf)] if buf.size else np.zeros(1, dtype=np.float32)
                if finite.size == 0:
                    finite = np.zeros(1, dtype=np.float32)
                qtls = np.linspace(0.0, 1.0, self.num_bins + 1)
                edges = np.unique(np.quantile(finite, qtls))
                if edges.size < 2:  # constant column -> a single bin
                    edges = np.concatenate([edges, edges + 1.0])
                self.num_edges[out] = edges
                # +1 local id for NaN values (its own bin)
                self.n_local[out] = edges.size - 1 + 1
                self._acc_num[out] = None
                del buf, finite

        self.vocab.build(self.out_columns, [self.n_local[c] for c in self.out_columns])
        if user_map is not None:
            self.vocab.user_map = dict(user_map)

    @property
    def total_vocab(self):
        return self.vocab.total

    def fit_user_map(self, df, user_col):
        """Build the user -> contiguous code map (call before finalize)."""
        acc = {}
        for val in df[user_col].value_counts(dropna=False).index:
            acc[canon(val)] = len(acc)
        return acc

    # ----------------------------------------------------------- transform
    def _transform_cat(self, series, out):
        mapping = self.cat_maps[out]
        base = self.vocab.bases[out]
        n = len(series)
        keys = series.map(canon).to_numpy(dtype=object)
        out_ids = np.fromiter(
            (mapping.get(k, -1) for k in keys), dtype=np.int64, count=n
        ).astype(np.int32)
        out_ids[out_ids >= 0] += base
        out_ids[out_ids < 0] = UNK_ID
        return out_ids

    def _transform_num(self, vals, out):
        edges = self.num_edges[out]
        base = self.vocab.bases[out]
        nan_mask = ~np.isfinite(vals)
        local = np.searchsorted(edges, vals, side="right") - 1
        local = np.clip(local, 0, edges.size - 2).astype(np.int32)
        local[nan_mask] = edges.size - 1  # the NaN bin
        return (local + base).astype(np.int32)

    def _transform_col(self, df, col):
        """Return a list of int32 arrays [n] for one original column."""
        kind = self.kinds[col]
        if kind == "cat":
            return [self._transform_cat(df[col], col)]
        if kind == "num":
            vals = self._as_float(df[col], col)
            if self._signed.get(col):
                # nuFormer-style: sign token + magnitude bin
                sign_out, bin_out = f"{col}#sign", f"{col}#bin"
                smap = self.cat_maps[sign_out]
                sbase = self.vocab.bases[sign_out]
                nan = ~np.isfinite(vals)
                sgn = np.where(vals < 0, smap.get("neg", 0), smap.get("pos", 1))
                sgn = sgn.astype(np.int32) + sbase
                sgn[nan] = UNK_ID
                bins = self._transform_num(np.abs(vals), bin_out)
                return [sgn, bins]
            return [self._transform_num(vals, col)]
        # dt
        d = self._as_dt(df[col], col)
        blocks = []
        for part in self.DT_SUBS:
            out = f"{col}#{part}"
            mapping = self.cat_maps[out]
            base = self.vocab.bases[out]
            vals = getattr(d.dt, "dayofweek" if part == "weekday" else part)
            keys = _int_keys(vals)
            ids = np.fromiter(
                (mapping.get(k, -1) for k in keys), dtype=np.int64, count=len(d)
            ).astype(np.int32)
            ids[ids >= 0] += base
            ids[ids < 0] = UNK_ID
            blocks.append(ids)
        blocks.append(self._transform_num(
            d.dt.year.to_numpy(dtype=np.float32), f"{col}#{self.DT_NUM}"))
        return blocks

    def transform(self, df, user_col=None, with_user=False):
        """Encode `df` (same feature columns as fit) into global ids.

        Returns
        -------
        encoded : np.ndarray int32 [n_rows, len(out_columns)]
        user_codes : np.ndarray int64 [n_rows]   (only if with_user)
        """
        missing = [c for c in self.columns if c not in df.columns]
        if missing:
            raise ValueError(f"columns missing from data: {missing}")

        blocks = []
        for col in self.columns:
            blocks += self._transform_col(df, col)
        assert len(blocks) == len(self.out_columns)
        encoded = np.stack(blocks, axis=1).astype(np.int32)

        if not with_user:
            return encoded

        user_keys = df[user_col].map(canon).to_numpy(dtype=object)
        umap = self.vocab.user_map
        next_code = len(umap)
        codes = np.empty(len(df), dtype=np.int64)
        for i, k in enumerate(user_keys):
            code = umap.get(k)
            if code is None:  # unseen user: fresh code (still groupable)
                code = next_code
                umap[k] = code
                next_code += 1
            codes[i] = code
        return encoded, codes

    # -------------------------------------------------------- serialization
    def get_state(self):
        return {
            "num_bins": self.num_bins,
            "min_freq": self.min_freq,
            "datetime_cols": self.datetime_cols,
            "decompose_dates": self.decompose_dates,
            "sign_split": self.sign_split,
            "categorical_cols": self.categorical_cols,
            "columns": self.columns,
            "kinds": self.kinds,
            "signed": self._signed,
            "out_columns": self.out_columns,
            "cat_maps": self.cat_maps,
            "num_edges": {c: np.asarray(e).tolist()
                          for c, e in self.num_edges.items()},
            "n_local": self.n_local,
            "vocab": self.vocab.get_state(),
        }

    def set_state(self, state):
        self.num_bins = state["num_bins"]
        self.min_freq = state["min_freq"]
        self.datetime_cols = list(state.get("datetime_cols") or [])
        self.decompose_dates = bool(state.get("decompose_dates", False))
        self.sign_split = bool(state.get("sign_split", False))
        self.categorical_cols = list(state.get("categorical_cols") or [])
        self.columns = list(state["columns"])
        self.kinds = dict(state["kinds"])
        self._signed = dict(state.get("signed") or {})
        self.out_columns = list(state.get("out_columns") or self.columns)
        self.cat_maps = {c: dict(m) for c, m in state["cat_maps"].items()}
        self.num_edges = {c: np.asarray(e) for c, e in state["num_edges"].items()}
        self.n_local = dict(state["n_local"])
        self.vocab = FieldVocab()
        self.vocab.set_state(state["vocab"])
