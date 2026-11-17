"""Sliding-window sample construction, memory-lean.

Windows are stored as a dense int32 matrix [n_windows, seq_len * ncols]
(in RAM, or in a disk memmap when bigger than `in_ram_limit` bytes).
"""

import os
import numpy as np


def _allocate(rows, cols, cache_dir, in_ram_limit, tag):
    nbytes = int(rows) * int(cols) * 4
    if nbytes <= in_ram_limit or cache_dir is None:
        return np.zeros((rows, cols), dtype=np.int32), False
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{tag}_int32.npy")
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.int32,
                                    shape=(rows, cols))
    return arr, True


def count_windows(user_counts, seq_len, stride):
    """n_windows per user: max(0, (T - seq_len) // stride + 1)."""
    counts = np.maximum(0, (np.asarray(user_counts) - seq_len) // stride + 1)
    return counts


def build_windows(encoded, user_codes, seq_len, stride,
                  cache_dir=None, in_ram_limit=256 * 1024 * 1024, verbose=False):
    """Build sliding windows over each user's ordered records.

    Parameters
    ----------
    encoded : int32 [n_rows, ncols], already sorted by user
    user_codes : int64 [n_rows], matching sort
    Returns
    -------
    windows   : int32 [n_windows, seq_len*ncols]
    meta      : dict with 'user' (int64 [n_windows]) and 'start' (int64)
    """
    n_rows, ncols = encoded.shape
    if n_rows == 0:
        raise ValueError("no rows to build windows from")

    # user boundaries in the sorted arrays
    change = np.empty(n_rows, dtype=bool)
    change[0] = True
    change[1:] = user_codes[1:] != user_codes[:-1]
    starts = np.flatnonzero(change)
    counts = np.diff(np.append(starts, n_rows))

    n_win = count_windows(counts, seq_len, stride)
    total = int(n_win.sum())
    dropped = int((n_win == 0).sum())
    if total == 0:
        raise ValueError(
            f"no windows could be built: every user has fewer than "
            f"{seq_len} records"
        )
    if verbose and dropped:
        print(f"[tabformer] {dropped} users with < {seq_len} records were skipped")

    windows, mmapped = _allocate(total, seq_len * ncols, cache_dir,
                                 in_ram_limit, tag="windows")
    win_user = np.empty(total, dtype=np.int64)
    win_start = np.empty(total, dtype=np.int64)

    w = 0
    for u_start, u_count, n_u in zip(starts, counts, n_win):
        if n_u == 0:
            continue
        offs = np.arange(n_u, dtype=np.int64) * stride + u_start
        # gather seq_len consecutive rows per offset
        idx = offs[:, None] + np.arange(seq_len, dtype=np.int64)[None, :]
        windows[w:w + n_u] = encoded[idx].reshape(n_u, seq_len * ncols)
        win_user[w:w + n_u] = user_codes[u_start]
        win_start[w:w + n_u] = offs
        w += n_u

    if mmapped:
        windows.flush()
    meta = {"user": win_user, "start": win_start, "mmapped": mmapped}
    return windows, meta


def batch_iterator(windows, indices, batch_size):
    """Yield (batch_idx_array, int64 tensor [bsz, seq_len, ncols])."""
    n = len(indices)
    for b in range(0, n, batch_size):
        sel = indices[b:b + batch_size]
        yield sel, windows[sel]  # numpy fancy indexing -> materializes batch
