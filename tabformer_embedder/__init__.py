"""TabFormer/nuFormer embeddings with a scikit-learn-like API (low-memory).

Example
-------
    from tabformer_embedder import TabFormerEmbedder

    emb = TabFormerEmbedder(user_col="User")     # arch="nuformer" (default)
    emb.fit(df)                    # causal next-token pretraining
    E = emb.transform(df)          # np.ndarray [n_windows, embedding_dim]

    # TabFormer's hierarchical MLM instead:
    emb = TabFormerEmbedder(user_col="User", arch="tabformer")

Works with any modern PyTorch (>= 2.0); does not require `transformers`.
"""

from .embedder import TabFormerEmbedder
from .model import TabFormerModel, NuFormerModel
from .vocab import PAD_ID, MASK_ID, UNK_ID, SEP_ID

__version__ = "0.2.1"
__all__ = ["TabFormerEmbedder", "TabFormerModel", "NuFormerModel",
           "PAD_ID", "MASK_ID", "UNK_ID", "SEP_ID"]
