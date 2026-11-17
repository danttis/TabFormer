from transformers.tokenization_utils import PreTrainedTokenizer


class TabFormerTokenizer(PreTrainedTokenizer):
    """Minimal field-token tokenizer for the GPT2 side of TabFormer.

    The real vocabulary lives in `dataset.vocab.Vocabulary` (built from the
    data); this tokenizer only tracks special tokens. Older transformers
    (e.g. 3.2.0) allowed subclassing `PreTrainedTokenizer` without
    implementing `get_vocab`; newer versions (>= 4.x) require it.
    """

    def __init__(
        self,
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        **kwargs,
    ):
        super().__init__(bos_token=bos_token, eos_token=eos_token, unk_token=unk_token, **kwargs)

    # --- minimal implementation required by transformers >= 4.x ---
    def get_vocab(self):
        vocab = {tok: idx for idx, tok in enumerate(self.all_special_tokens)}
        return vocab

    @property
    def vocab_size(self):
        return len(self.get_vocab())

    def save_vocabulary(self, save_directory, filename_prefix=None):
        # no vocabulary files to save (special tokens live in tokenizer_config)
        return ()

    def _tokenize(self, text):
        return text.split()

    def _convert_token_to_id(self, token):
        return self.get_vocab().get(token, self.unk_token_id)
