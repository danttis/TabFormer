"""TabFormerEmbedder: sklearn-style .fit() / .transform() over TabFormer/nuFormer.

    from tabformer_embedder import TabFormerEmbedder

    emb = TabFormerEmbedder(user_col="User")   # arch="nuformer" by default
    emb.fit(df)                                # causal next-token pretraining
    E  = emb.transform(df)                     # [n_windows, embedding_dim]
    E  = emb.transform(df, pooling="last")     # nuFormer's user representation

Two architectures:
  * arch="nuformer" (default, arXiv:2507.23267): flat causal transformer,
    no positional embeddings (NoPE), next-token prediction, [SEP] between
    transactions, dates decomposed into month/day/weekday/hour tokens,
    numeric columns with negatives split into sign + magnitude bin.
  * arch="tabformer" (arXiv:2011.01884): hierarchical MLM -- field-level
    transformer + transaction-level transformer.

Design for machines with little memory:
  * tokens stored as int32 (RAM or disk memmap above ~256 MB);
  * small model by default (hidden 128, 2 layers);
  * loss computed only at trained positions, in chunks (bounded peak even
    with large vocabularies);
  * gradient accumulation instead of large batches;
  * CSV input is read in chunks (two passes), never fully in RAM.
"""

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .vocab import canon, SEP_ID
from .preprocess import TabularPreprocessor
from .data import build_windows, batch_iterator
from .model import TabFormerModel, NuFormerModel

__all__ = ["TabFormerEmbedder"]

_RAM_LIMIT_BYTES = 256 * 1024 * 1024  # above this, arrays go to disk memmap


class TabFormerEmbedder:
    """Tabular embeddings with a scikit-learn-like API.

    Parameters
    ----------
    user_col : column that groups rows into sequences (e.g. "User").
    target_col : optional label column to EXCLUDE from the embeddings.
    arch : "nuformer" (flat causal + next-token, default) or
        "tabformer" (hierarchical + MLM).
    datetime_cols : columns to parse as dates when reading CSV.
    decompose_dates : nuFormer-style date tokens (month/day/weekday/hour
        + binned year). None = auto: on for nuformer, off for tabformer.
    sign_split : nuFormer-style sign + magnitude tokens for numeric
        columns containing negatives. None = auto (same rule).
    seq_len, stride : sliding window over each user's history.
    num_bins : quantile bins for numeric columns.
    min_freq : categories seen fewer times become [UNK].
    embedding_dim : transformer hidden size (= output dim).
    field_hidden : per-field embedding size (tabformer arch only).
    n_layers, n_heads : transformer layers/heads.
    field_layers, field_heads : field-level transformer (tabformer only).
    batch_size, grad_accum : effective batch = batch_size * grad_accum.
    epochs, lr, weight_decay, mlm_prob, val_frac : training knobs
        (mlm_prob applies to the tabformer arch only).
    device : "auto" (cuda if available), "cpu" or "cuda".
    cache_dir : where big token matrices are memmapped.
    """

    def __init__(self,
                 user_col="User",
                 target_col=None,
                 arch="nuformer",
                 datetime_cols=None,
                 decompose_dates=None,
                 sign_split=None,
                 categorical_cols=None,
                 seq_len=10,
                 stride=5,
                 num_bins=10,
                 min_freq=1,
                 embedding_dim=128,
                 field_hidden=32,
                 n_layers=2,
                 n_heads=4,
                 field_layers=1,
                 field_heads=4,
                 feedforward_mult=2,
                 dropout=0.1,
                 batch_size=64,
                 grad_accum=1,
                 epochs=5,
                 lr=3e-4,
                 weight_decay=0.01,
                 mlm_prob=0.15,
                 val_frac=0.1,
                 device="auto",
                 seed=42,
                 verbose=True,
                 cache_dir=None):
        if arch not in ("nuformer", "tabformer"):
            raise ValueError(f"unknown arch '{arch}'")
        self.user_col = user_col
        self.target_col = target_col
        self.arch = arch
        self.datetime_cols = list(datetime_cols) if datetime_cols else None
        self.decompose_dates = decompose_dates
        self.sign_split = sign_split
        self.categorical_cols = list(categorical_cols) if categorical_cols else None
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.num_bins = int(num_bins)
        self.min_freq = int(min_freq)
        self.embedding_dim = int(embedding_dim)
        self.field_hidden = int(field_hidden)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.field_layers = int(field_layers)
        self.field_heads = int(field_heads)
        self.feedforward_mult = int(feedforward_mult)
        self.dropout = float(dropout)
        self.batch_size = int(batch_size)
        self.grad_accum = int(grad_accum)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.mlm_prob = float(mlm_prob)
        self.val_frac = float(val_frac)
        self.device = device
        self.seed = int(seed)
        self.verbose = bool(verbose)
        self.cache_dir = cache_dir

        self.prep_ = None
        self.model_ = None
        self._ncols_row = None
        self._user_counts = {}
        self.history_ = []

    # ------------------------------------------------------------ internals
    def _log(self, *a):
        if self.verbose:
            print("[tabformer]", *a)

    def _device(self):
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _set_seeds(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def _new_prep(self):
        decompose = (self.arch == "nuformer") if self.decompose_dates is None \
            else bool(self.decompose_dates)
        sign = (self.arch == "nuformer") if self.sign_split is None \
            else bool(self.sign_split)
        return TabularPreprocessor(num_bins=self.num_bins,
                                   min_freq=self.min_freq,
                                   datetime_cols=self.datetime_cols,
                                   decompose_dates=decompose,
                                   sign_split=sign,
                                   categorical_cols=self.categorical_cols)

    def _feature_cols(self, columns):
        return [c for c in columns
                if c != self.user_col and c != self.target_col]

    def _add_sep(self, encoded):
        """nuFormer: prepend a [SEP] column so every transaction starts with
        its boundary marker and the LAST token is a real field token."""
        if self.arch != "nuformer":
            return encoded
        sep = np.full((encoded.shape[0], 1), SEP_ID, dtype=np.int32)
        return np.concatenate([sep, encoded], axis=1)

    def _view(self, flat_batch, device):
        """Flat int32 window row [B, seq_len*ncols_row] -> model input."""
        x = torch.from_numpy(np.ascontiguousarray(flat_batch)).to(device).long()
        if self.arch == "tabformer":
            x = x.view(x.shape[0], self.seq_len, -1)
        # nuformer: keep flat [B, L]
        return x

    def _build_model(self, vocab_size, ncols_row):
        common = dict(n_layers=self.n_layers, n_heads=self.n_heads,
                      feedforward_mult=self.feedforward_mult,
                      dropout=self.dropout)
        if self.arch == "nuformer":
            return NuFormerModel(vocab_size, hidden=self.embedding_dim, **common)
        return TabFormerModel(
            vocab_size=vocab_size,
            ncols=ncols_row,
            seq_len=self.seq_len,
            field_hidden=self.field_hidden,
            hidden=self.embedding_dim,
            field_layers=self.field_layers,
            field_heads=self.field_heads,
            mlm_prob=self.mlm_prob,
            **common,
        )

    # ----------------------------------------------------------------- fit
    def fit(self, data, chunksize=200_000):
        """Fit on a pandas DataFrame or a CSV path (streamed in chunks).

        Returns self.
        """
        self._set_seeds()
        if isinstance(data, (str, Path)):
            encoded, user_codes = self._fit_csv(str(data), chunksize)
        else:
            encoded, user_codes = self._fit_df(data)
        return self._train(encoded, user_codes)

    def fit_transform(self, data, **kw):
        return self.fit(data).transform(data, **kw)

    def _fit_df(self, df):
        if self.user_col not in df.columns:
            raise ValueError(f"user_col '{self.user_col}' not in data")
        feats = self._feature_cols(df.columns)
        prep = self._new_prep()

        for val, cnt in df[self.user_col].value_counts(dropna=False).items():
            k = canon(val)
            self._user_counts[k] = self._user_counts.get(k, 0) + int(cnt)

        prep.accumulate(df[feats])
        prep.finalize(user_map={
            k: i for i, k in enumerate(sorted(self._user_counts))})
        self.prep_ = prep

        encoded, user_codes = prep.transform(df[feats + [self.user_col]],
                                            user_col=self.user_col, with_user=True)
        return self._add_sep(encoded), user_codes

    def _fit_csv(self, path, chunksize):
        prep = self._new_prep()
        self._user_counts = {}
        n_rows = 0
        feats = None
        parse_dates = self.datetime_cols
        # pass 1: accumulate vocab stats (bounded memory)
        for chunk in pd.read_csv(path, chunksize=chunksize, parse_dates=parse_dates):
            if feats is None:
                header = list(chunk.columns)
                if self.user_col not in header:
                    raise ValueError(f"user_col '{self.user_col}' not in data")
                feats = self._feature_cols(header)
            for val, cnt in chunk[self.user_col].value_counts(dropna=False).items():
                k = canon(val)
                self._user_counts[k] = self._user_counts.get(k, 0) + int(cnt)
            prep.accumulate(chunk[feats])
            n_rows += len(chunk)
        prep.finalize(user_map={
            k: i for i, k in enumerate(sorted(self._user_counts))})
        self.prep_ = prep
        self._log(f"vocab size: {prep.total_vocab} tokens, rows: {n_rows}, "
                  f"fields: {len(prep.out_columns)}")

        # pass 2: encode into a preallocated int32 matrix (memmap if big)
        width = len(prep.out_columns) + (1 if self.arch == "nuformer" else 0)
        memmap = width * n_rows * 4 > _RAM_LIMIT_BYTES and self.cache_dir is not None
        if memmap:
            os.makedirs(self.cache_dir, exist_ok=True)
            encoded = np.lib.format.open_memmap(
                os.path.join(self.cache_dir, "encoded_int32.npy"),
                mode="w+", dtype=np.int32, shape=(n_rows, width))
        else:
            encoded = np.empty((n_rows, width), dtype=np.int32)
        user_codes = np.empty(n_rows, dtype=np.int64)

        i = 0
        for chunk in pd.read_csv(path, chunksize=chunksize, parse_dates=parse_dates):
            j = i + len(chunk)
            enc, user_codes[i:j] = prep.transform(
                chunk[prep.columns + [self.user_col]],
                user_col=self.user_col, with_user=True)
            if self.arch == "nuformer":
                encoded[i:j, 0] = SEP_ID
                encoded[i:j, 1:] = enc
            else:
                encoded[i:j] = enc
            i = j
        if memmap:
            encoded.flush()
        return encoded, user_codes

    def _train(self, encoded, user_codes):
        prep = self.prep_
        # sort by user (stable) so windows don't mix users
        order = np.argsort(user_codes, kind="stable")
        encoded = encoded[order]
        user_codes = user_codes[order]
        self._ncols_row = encoded.shape[1]

        windows, meta = build_windows(
            encoded, user_codes, self.seq_len, self.stride,
            cache_dir=self.cache_dir, in_ram_limit=_RAM_LIMIT_BYTES,
            verbose=self.verbose)
        n_windows = windows.shape[0]
        self._log(f"arch: {self.arch} | windows: {n_windows} x "
                  f"{self.seq_len} x {self._ncols_row} fields "
                  f"({self.seq_len * self._ncols_row} tokens/window)")

        model = self._build_model(prep.total_vocab, self._ncols_row)
        device = self._device()
        model.to(device)
        self._log(f"model params: "
                  f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

        # train/val split at window level
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(n_windows)
        n_val = max(1, int(self.val_frac * n_windows)) if self.val_frac > 0 else 0
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        opt = torch.optim.AdamW(model.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
        use_amp = device.type == "cuda"

        best_val = float("inf")
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        self.history_ = []

        for epoch in range(self.epochs):
            model.train()
            perm = rng.permutation(len(train_idx))
            train_idx = train_idx[perm]
            opt.zero_grad(set_to_none=True)
            running = 0.0
            n_steps = 0
            accum = 0

            for sel, batch in batch_iterator(windows, train_idx, self.batch_size):
                x = self._view(batch, device)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    # the model's loss backprops internally with the AMP
                    # scale (and grad-accum divisor) already folded in
                    loss = model.training_loss(
                        x, scale=scaler.get_scale() / self.grad_accum)
                running += float(loss)
                n_steps += 1
                accum += 1
                if accum == self.grad_accum:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    accum = 0

            val_loss = self._validate(model, windows, val_idx, device)
            self.history_.append({"epoch": epoch + 1,
                                  "train_loss": running / max(n_steps, 1),
                                  "val_loss": val_loss})
            self._log(f"epoch {epoch+1}/{self.epochs}  "
                      f"train {running / max(n_steps,1):.4f}  val {val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        self.model_ = model.to(device)
        self._log("fit done")
        return self

    def _validate(self, model, windows, val_idx, device):
        if len(val_idx) == 0:
            return float("nan")
        model.eval()
        total, steps = 0.0, 0
        for sel, batch in batch_iterator(windows, val_idx, self.batch_size):
            x = self._view(batch, device)
            total += float(model.eval_loss(x))
            steps += 1
        return total / max(steps, 1)

    # ---------------------------------------------------------- transform
    def transform(self, data, batch_size=None, pooling="mean",
                  return_metadata=False, out_path=None, chunksize=200_000):
        """Generate embeddings. Returns np.ndarray [n_windows, embedding_dim].

        pooling: "mean" | "last" | "max" | "flatten" (all tokens * dim).
            "last" is nuFormer's user representation (final token state).
        out_path: optional .npy path -- result is written to a disk memmap
                  (useful when the full matrix does not fit in RAM).
        return_metadata: also return a DataFrame with user / start of each
                  window (users unseen during fit get NaN).
        """
        if self.model_ is None or self.prep_ is None:
            raise RuntimeError("call .fit() before .transform()")
        model = self.model_
        prep = self.prep_
        batch_size = batch_size or self.batch_size
        device = next(model.parameters()).device

        encoded, user_codes = self._encode_new(data, chunksize)
        windows, meta = build_windows(
            encoded, user_codes, self.seq_len, self.stride,
            cache_dir=self.cache_dir, in_ram_limit=_RAM_LIMIT_BYTES)

        n_windows = windows.shape[0]
        # tokens per window along the pooling axis
        if self.arch == "tabformer":
            tokens = self.seq_len
        else:
            tokens = self.seq_len * encoded.shape[1]
        if pooling == "flatten":
            out_dim = self.embedding_dim * tokens
        else:
            out_dim = self.embedding_dim
        if out_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            emb = np.lib.format.open_memmap(out_path, mode="w+",
                                            dtype=np.float32,
                                            shape=(n_windows, out_dim))
        else:
            emb = np.empty((n_windows, out_dim), dtype=np.float32)

        model.eval()
        all_idx = np.arange(n_windows)
        with torch.no_grad():
            w = 0
            for sel, batch in batch_iterator(windows, all_idx, batch_size):
                x = self._view(batch, device)
                hidden = model.embed_tokens(x)      # [B, tokens, H]
                if pooling == "mean":
                    pooled = hidden.mean(dim=1)
                elif pooling == "last":
                    pooled = hidden[:, -1]
                elif pooling == "max":
                    pooled = hidden.max(dim=1).values
                elif pooling == "flatten":
                    pooled = hidden.reshape(hidden.shape[0], -1)
                else:
                    raise ValueError(f"unknown pooling '{pooling}'")
                emb[w:w + len(sel)] = pooled.detach().cpu().numpy().astype(np.float32)
                w += len(sel)
        if out_path is not None:
            emb.flush()

        if return_metadata:
            code_to_user = {v: k for k, v in prep.vocab.user_map.items()}
            users = np.array([code_to_user.get(c, np.nan) for c in meta["user"]])
            md = pd.DataFrame({"user": users, "start": meta["start"]})
            return emb, md
        return emb

    def embed(self, data, **kw):
        """Alias for transform()."""
        return self.transform(data, **kw)

    def _encode_new(self, data, chunksize):
        prep = self.prep_
        if isinstance(data, (str, Path)):
            n_rows = sum(len(c) for c in pd.read_csv(str(data), chunksize=chunksize,
                                                     parse_dates=self.datetime_cols))
            ncols = len(prep.out_columns)
            width = ncols + (1 if self.arch == "nuformer" else 0)
            encoded = np.empty((n_rows, width), dtype=np.int32)
            user_codes = np.empty(n_rows, dtype=np.int64)
            i = 0
            for chunk in pd.read_csv(str(data), chunksize=chunksize,
                                      parse_dates=self.datetime_cols):
                j = i + len(chunk)
                enc, user_codes[i:j] = prep.transform(
                    chunk[prep.columns + [self.user_col]],
                    user_col=self.user_col, with_user=True)
                if self.arch == "nuformer":
                    encoded[i:j, 0] = SEP_ID
                    encoded[i:j, 1:] = enc
                else:
                    encoded[i:j] = enc
                i = j
        else:
            if self.user_col not in data.columns:
                raise ValueError(f"user_col '{self.user_col}' not in data")
            encoded, user_codes = prep.transform(
                data[prep.columns + [self.user_col]],
                user_col=self.user_col, with_user=True)
            encoded = self._add_sep(encoded)
        return encoded, user_codes

    # ------------------------------------------------------- save / load
    def save(self, path):
        """Save model + vocab to a single file (.pt)."""
        if self.model_ is None:
            raise RuntimeError("nothing to save: call .fit() first")
        payload = {
            "format": "tabformer_embedder",
            "version": 2,
            "params": {k: getattr(self, k) for k in (
                "user_col", "target_col", "arch", "datetime_cols",
                "decompose_dates", "sign_split", "categorical_cols",
                "seq_len", "stride", "num_bins", "min_freq",
                "embedding_dim", "field_hidden", "n_layers",
                "n_heads", "field_layers", "field_heads", "feedforward_mult",
                "dropout", "mlm_prob", "verbose")},
            "prep_state": self.prep_.get_state(),
            "model_state": {k: v.cpu() for k, v in self.model_.state_dict().items()},
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path, device="auto", verbose=True):
        """Load a saved embedder: TabFormerEmbedder.load('model.pt')."""
        payload = torch.load(path, map_location="cpu", weights_only=False)
        params = payload["params"]
        obj = cls(user_col=params["user_col"], verbose=verbose, **{
            k: v for k, v in params.items()
            if k not in ("user_col", "verbose")})
        obj.device = device
        prep = TabularPreprocessor(num_bins=params["num_bins"],
                                   min_freq=params["min_freq"])
        prep.set_state(payload["prep_state"])
        obj.prep_ = prep

        ncols_row = len(prep.out_columns) + (1 if obj.arch == "nuformer" else 0)
        obj._ncols_row = ncols_row
        model = obj._build_model(prep.total_vocab, ncols_row)
        model.load_state_dict({k: v for k, v in payload["model_state"].items()})
        if device == "auto":
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            dev = torch.device(device)
        obj.model_ = model.to(dev)
        return obj
