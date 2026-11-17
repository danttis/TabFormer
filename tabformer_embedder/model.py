"""Hierarchical TabFormer encoder in pure PyTorch (no `transformers` needed).

Architecture (the paper's hierarchical variant, shrunk for low-RAM machines):

    input [B, S, C]  (S = seq_len transactions, C = ncols fields)
      -> nn.Embedding(vocab, field_hidden)          per-token field embedding
      -> 1-layer transformer over the C fields      (field-level mixing)
      -> concat fields -> Linear -> hidden           (row representation)
      -> + learned positional embedding (S)
      -> n_layers transformer over S rows           (transaction-level mixing)
      -> hidden states [B, S, hidden]                (the embeddings)

    MLM head (training only), mirroring the original TabFormer head: the row
    state is projected back to per-field slices, and a shared decoder maps
    each slice to that field's vocabulary:

      hidden [B,S,H] -> proj -> [B,S,C,FH] -> shared decoder -> vocab logits
"""

import torch
from torch import nn
import torch.nn.functional as F

from .vocab import PAD_ID, MASK_ID, N_SPECIAL


class TabFormerModel(nn.Module):
    def __init__(self, vocab_size, ncols, seq_len,
                 field_hidden=32, hidden=128,
                 n_layers=2, n_heads=4,
                 field_layers=1, field_heads=4,
                 feedforward_mult=2, dropout=0.1,
                 mlm_prob=0.15):
        super().__init__()
        if field_hidden % field_heads != 0:
            raise ValueError(f"field_hidden ({field_hidden}) must be divisible "
                             f"by field_heads ({field_heads})")
        if hidden % n_heads != 0:
            raise ValueError(f"hidden ({hidden}) must be divisible "
                             f"by n_heads ({n_heads})")

        self.ncols = ncols
        self.seq_len = seq_len
        self.hidden_size = hidden
        self.field_hidden = field_hidden
        self.vocab_size = vocab_size
        self.mlm_prob = float(mlm_prob)

        self.field_emb = nn.Embedding(vocab_size, field_hidden, padding_idx=PAD_ID)
        self.field_tf = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(field_hidden, field_heads,
                                        field_hidden * feedforward_mult,
                                        dropout, batch_first=True),
            num_layers=field_layers)
        self.proj = nn.Linear(field_hidden * ncols, hidden)
        self.pos_emb = nn.Embedding(seq_len, hidden)
        self.row_tf = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden, n_heads,
                                        hidden * feedforward_mult,
                                        dropout, batch_first=True),
            num_layers=n_layers)

        # MLM head: row state -> per-field slices -> shared decoder
        self.head_proj = nn.Linear(hidden, ncols * field_hidden)
        self.head = nn.Sequential(
            nn.Linear(field_hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, vocab_size),
        )

        self.mask_id = MASK_ID

    # ------------------------------------------------------------ encoding
    def encode(self, input_ids):
        """input_ids [B, S, C] (long) -> hidden states [B, S, hidden]."""
        bsz, seqlen, ncols = input_ids.shape
        if ncols != self.ncols:
            raise ValueError(f"expected {self.ncols} columns, got {ncols}")

        x = self.field_emb(input_ids)                       # [B,S,C,FH]
        x = x.reshape(bsz * seqlen, ncols, self.field_hidden)
        x = self.field_tf(x)                                # field-level
        x = x.reshape(bsz, seqlen, ncols * self.field_hidden)
        x = self.proj(x)                                    # [B,S,H]

        pos = torch.arange(seqlen, device=input_ids.device)
        x = x + self.pos_emb(pos)[None, :, :]
        hidden = self.row_tf(x)                             # [B,S,H]
        return hidden

    def token_states(self, hidden):
        """Row hidden [B,S,H] -> per-field token reps [B*S*C, FH] (has grad)."""
        bsz, seqlen, _ = hidden.shape
        tok = self.head_proj(hidden).view(bsz, seqlen, self.ncols, self.field_hidden)
        return tok.reshape(bsz * seqlen * self.ncols, self.field_hidden)

    def forward(self, input_ids, labels=None):
        """labels: [B, S, C] with -100 on unmasked positions."""
        hidden = self.encode(input_ids)
        out = {"hidden": hidden}
        if labels is not None:
            tok = self.token_states(hidden)
            out["loss"] = chunked_mlm_loss(self.head, tok, labels.reshape(-1))
        return out

    # -------------------------------------------------- loss interface
    def training_loss(self, x, scale=1.0):
        """x: [B, S, C] raw token ids. Backprops internally, returns loss."""
        x_in, labels = apply_mlm_mask(x, self.mask_id, self.mlm_prob,
                                      vocab_size=self.vocab_size)
        hidden = self.encode(x_in)
        tok = self.token_states(hidden)
        return chunked_mlm_loss(self.head, tok, labels.reshape(-1), scale=scale)

    @torch.no_grad()
    def eval_loss(self, x):
        x_in, labels = apply_mlm_mask(x, self.mask_id, self.mlm_prob,
                                      vocab_size=self.vocab_size)
        hidden = self.encode(x_in)
        tok = self.token_states(hidden)
        return eval_mlm_loss(self.head, tok, labels.reshape(-1))

    def embed_tokens(self, x):
        """x: [B, S, C] -> per-row hidden states [B, S, hidden]."""
        return self.encode(x)


class NuFormerModel(nn.Module):
    """nuFormer-style flat causal transformer (arXiv:2507.23267).

    * decoder-only: each token of a window (fields + [SEP] markers) goes
      directly into a causal transformer -- no field-level hierarchy;
    * NoPE: no positional embeddings (causal masking already encodes
      order, and generalizes to longer sequences);
    * pretraining objective: next-token prediction over the whole vocab;
    * sequence representation: hidden state of the last token
      (pooling="last" in TabFormerEmbedder.transform).
    """

    def __init__(self, vocab_size, hidden=128,
                 n_layers=2, n_heads=4,
                 feedforward_mult=2, dropout=0.1):
        super().__init__()
        if hidden % n_heads != 0:
            raise ValueError(f"hidden ({hidden}) must be divisible "
                             f"by n_heads ({n_heads})")
        self.vocab_size = vocab_size
        self.hidden_size = hidden
        self.emb = nn.Embedding(vocab_size, hidden, padding_idx=PAD_ID)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden, n_heads,
                                        hidden * feedforward_mult,
                                        dropout, batch_first=True)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(hidden, vocab_size)
        self._causal_cache = {}

    def _causal_mask(self, seqlen, device):
        key = (seqlen, str(device))
        if key not in self._causal_cache:
            self._causal_cache[key] = torch.triu(
                torch.ones(seqlen, seqlen, dtype=torch.bool, device=device),
                diagonal=1)
        return self._causal_cache[key]

    def encode(self, input_ids):
        """input_ids [B, L] (long) -> hidden states [B, L, hidden] (causal)."""
        h = self.emb(input_ids)
        mask = self._causal_mask(input_ids.shape[1], input_ids.device)
        for layer in self.layers:
            # torch >= 2.4 renamed src_mask -> attn_mask; pass positionally
            h = layer(h, mask, None, True)
        return h

    def training_loss(self, x, scale=1.0):
        """x: [B, L] raw token ids. Next-token CE, backprops internally."""
        h = self.encode(x)
        reps = h[:, :-1].reshape(-1, self.hidden_size)
        targets = x[:, 1:].reshape(-1)
        return chunked_mlm_loss(self.head, reps, targets, scale=scale)

    @torch.no_grad()
    def eval_loss(self, x):
        h = self.encode(x)
        reps = h[:, :-1].reshape(-1, self.hidden_size)
        targets = x[:, 1:].reshape(-1)
        return eval_mlm_loss(self.head, reps, targets)

    def embed_tokens(self, x):
        """x: [B, L] -> per-token hidden states [B, L, hidden]."""
        return self.encode(x)


# ---------------------------------------------------------------- MLM utils
def apply_mlm_mask(input_ids, mask_id, mlm_prob=0.15, vocab_size=None,
                   generator=None):
    """80/10/10 masking. Returns (masked_inputs, labels with -100)."""
    labels = input_ids.clone()
    masked = torch.rand(input_ids.shape, generator=generator,
                        device=input_ids.device) < mlm_prob
    labels[~masked] = -100

    inputs = input_ids.clone()
    probs = torch.rand(input_ids.shape, generator=generator,
                       device=input_ids.device)
    replace_mask = masked & (probs < 0.8)
    inputs[replace_mask] = mask_id

    if vocab_size is not None:
        random_mask = masked & (probs >= 0.9)
        rand_tok = torch.randint(N_SPECIAL, vocab_size, input_ids.shape,
                                 generator=generator, device=input_ids.device)
        inputs[random_mask] = rand_tok[random_mask]
    return inputs, labels


def _last_linear_out(head):
    for m in reversed(list(head.modules())):
        if isinstance(m, nn.Linear):
            return m.out_features
    raise ValueError("head has no Linear layer")


def _masked_positions(flat_labels):
    pos = (flat_labels != -100).nonzero(as_tuple=True)[0]
    return pos, flat_labels.index_select(0, pos)


def chunked_mlm_loss(head, tok_reps, flat_labels, scale=1.0,
                     max_chunk_elems=32_000_000):
    """Memory-bounded MLM cross-entropy (runs backward internally).

    Logits are computed only for masked positions and in chunks; the hidden
    gradients are collected chunk by chunk so a chunk's logits are never all
    retained (peak activation ~= chunk x vocab floats, ~128MB by default).

    Gradients are accumulated into `.grad` (pre-multiplied by `scale`, e.g.
    AMP loss-scale / grad-accum), so the caller must NOT call backward again.
    Returns the *unscaled* loss value as a detached tensor.
    """
    pos, tgt = _masked_positions(flat_labels)
    n_masked = pos.numel()
    if n_masked == 0:
        return tok_reps.sum().detach() * 0.0  # no-op, grads stay None

    hm = tok_reps.index_select(0, pos)                  # [M, FH]
    vocab = _last_linear_out(head)
    chunk = max(64, int(max_chunk_elems // max(vocab, 1)))

    head_params = [p for p in head.parameters() if p.requires_grad]
    h_grad = torch.zeros_like(hm)
    total = hm.new_zeros(())
    for i in range(0, n_masked, chunk):
        hi = hm[i:i + chunk]
        logits = head(hi)
        li = F.cross_entropy(logits, tgt[i:i + chunk], reduction="sum") / n_masked
        # gradients of this chunk wrt the slice and the head weights;
        # autograd.grad frees the chunk's logits right after (bounded memory)
        grads = torch.autograd.grad(li, (hi, *head_params))
        h_grad[i:i + chunk] = grads[0] * scale
        for p, g in zip(head_params, grads[1:]):
            g = g * scale
            if p.grad is None:
                p.grad = g
            else:
                p.grad += g
        total = total + li.detach()
        del logits, li, grads
    hm.backward(gradient=h_grad)   # single pass back to the encoders
    return total


@torch.no_grad()
def eval_mlm_loss(head, tok_reps, flat_labels, max_chunk_elems=32_000_000):
    """Validation loss: same chunking, no gradient bookkeeping needed."""
    pos, tgt = _masked_positions(flat_labels)
    n_masked = pos.numel()
    if n_masked == 0:
        return tok_reps.new_zeros(())
    hm = tok_reps.index_select(0, pos)
    vocab = _last_linear_out(head)
    chunk = max(64, int(max_chunk_elems // max(vocab, 1)))
    total = hm.new_zeros(())
    for i in range(0, n_masked, chunk):
        logits = head(hm[i:i + chunk])
        total = total + F.cross_entropy(logits, tgt[i:i + chunk], reduction="sum")
    return total / n_masked
