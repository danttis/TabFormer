"""Field-aware vocabulary shared by all columns.

Layout of global ids:
    0 -> [PAD]
    1 -> [MASK]
    2 -> [UNK]
    3 -> [SEP]   (transaction separator, nuFormer-style flat arch)
    4.. -> column vocabularies, concatenated in the order of `out_columns`
"""

import numpy as np

PAD_ID = 0
MASK_ID = 1
UNK_ID = 2
SEP_ID = 3
N_SPECIAL = 4

SPECIAL_TOKENS = ["[PAD]", "[MASK]", "[UNK]", "[SEP]"]


class FieldVocab:
    """Holds, for each column, the mapping value -> global token id."""

    def __init__(self):
        self.columns = []          # ordered feature column names
        self.bases = {}            # col -> first global id of that column
        self.n_local = {}          # col -> number of local ids for that column
        self.total = N_SPECIAL     # total vocabulary size (incl. specials)
        self.user_map = {}         # canon user value -> contiguous user code

    def build(self, columns, per_col_sizes):
        self.columns = list(columns)
        self.bases, self.n_local = {}, {}
        base = N_SPECIAL
        for col, size in zip(columns, per_col_sizes):
            self.bases[col] = base
            self.n_local[col] = int(size)
            base += int(size)
        self.total = base

    def field_of_token(self, global_id):
        """Return the column name a global id belongs to (or 'SPECIAL')."""
        if global_id < N_SPECIAL:
            return "SPECIAL"
        for col in self.columns:
            if global_id < self.bases[col] + self.n_local[col]:
                return col
        return "SPECIAL"

    def get_state(self):
        return {
            "columns": self.columns,
            "bases": self.bases,
            "n_local": self.n_local,
            "total": self.total,
            "user_map": self.user_map,
        }

    def set_state(self, state):
        self.columns = list(state["columns"])
        self.bases = dict(state["bases"])
        self.n_local = dict(state["n_local"])
        self.total = int(state["total"])
        self.user_map = dict(state["user_map"])


def canon(value):
    """Canonical string key for a categorical value (NaN-aware)."""
    try:
        if value is None or np.isnan(value):
            return "__nan__"
    except (TypeError, ValueError):
        # non numeric scalars (str, etc.) -- np.isnan raises
        if value is None:
            return "__nan__"
    if isinstance(value, (np.floating, float)):
        f = float(value)
        if f != f:  # NaN caught above, but be safe
            return "__nan__"
        if f.is_integer():
            return str(int(f))
        return repr(f)
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.bool_, bool)):
        return "True" if value else "False"
    return str(value)
