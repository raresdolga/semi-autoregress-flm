"""Turning raw text into token ids.

Two stages, applied in this order: a per-corpus detokenizer undoes the
tokenization baked into a published corpus, then a tokenizer maps the cleaned
text to ids.
"""

import re
import typing

import tokenizers
import transformers


# ---------------------------------------------------------------------------
# Detokenizers
# ---------------------------------------------------------------------------


def wt_detokenizer(string):
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string


def ptb_detokenizer(x):
    x = x.replace(" 's", "'s")
    x = x.replace("s ' ", "s' ")
    x = x.replace(" n't", "n't")
    x = x.replace(" \n ", "\n")
    x = x.replace("\\/", "/")
    for _ in range(10):
        x = x.replace(" N ", " 1 ")
    x = x.replace("$ 1", "$1")
    x = x.replace("# 1", "#1")
    x = x.replace("<unk>", "?")
    return x


def lm1b_detokenizer(x):
    x = x.replace("http : / / ", "http://")
    x = x.replace("https : / / ", "https://")
    x = re.sub(r" \'(\w+)", r"'\1", x)
    x = re.sub(r" (\w+) \. ", r" \1. ", x)
    x = re.sub(r" (\w+) \.$", r" \1.", x)
    x = x.replace(" ? ", "? ")
    x = re.sub(r" \?$", "?", x)
    x = x.replace(" ! ", "! ")
    x = re.sub(r" \!$", "!", x)
    x = x.replace(" , ", ", ")
    x = x.replace(" : ", ": ")
    x = x.replace(" ; ", "; ")
    x = x.replace(" / ", "/")
    x = re.sub(r"\" ([^\"]+) \"", r'"\1"', x)
    x = re.sub(r"\' ([^\']+) \'", r"'\1'", x)
    x = re.sub(r"\( ([^\(\)]+) \)", r"(\1)", x)
    x = re.sub(r"\[ ([^\[\]]+) \]", r"[\1]", x)
    x = x.replace("$ ", "$")
    x = x.replace("£ ", "£")
    return x


def lambada_detokenizer(text):
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    return "\n" + text.strip()


def scientific_papers_detokenizer(x):
    x = wt_detokenizer(x)
    x = lm1b_detokenizer(x)
    return x


DETOKENIZERS = {
    "ptb": ptb_detokenizer,
    "lm1b": lm1b_detokenizer,
    "lambada": lambada_detokenizer,
}


def detokenizer_for(dataset_name):
    """The detokenizer for `dataset_name`, or None if it needs none."""
    if dataset_name.startswith("wikitext"):
        return wt_detokenizer
    if dataset_name.startswith("scientific_papers"):
        return scientific_papers_detokenizer
    return DETOKENIZERS.get(dataset_name)


# ---------------------------------------------------------------------------
# Tokenizers
# ---------------------------------------------------------------------------


class SyntheticTokenizer(transformers.PreTrainedTokenizer):
    def __init__(
        self,
        vocab_size,
        bos_token="[BOS]",
        eos_token="[EOS]",
        sep_token=None,
        cls_token=None,
        pad_token=None,
        mask_token=None,
        unk_token=None,
        **kwargs,
    ):
        self.tokens = []

        for i in range(vocab_size - 2):
            # appending space for readability
            self.tokens.append(str(i) + ", ")

        self._vocab_str_to_int = {
            "[BOS]": vocab_size - 2,
            "[EOS]": vocab_size - 1,
            **{ch: i for i, ch in enumerate(self.tokens)},
        }

        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str, **kwargs) -> typing.List[str]:
        return list(text.lower())

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def get_vocab(self) -> typing.Dict[str, int]:
        return self._vocab_str_to_int


class Text8Tokenizer(transformers.PreTrainedTokenizer):
    def __init__(
        self,
        bos_token="[BOS]",
        eos_token="[EOS]",
        sep_token="[SEP]",
        cls_token="[CLS]",
        pad_token="[PAD]",
        mask_token="[MASK]",
        unk_token="[UNK]",
        **kwargs,
    ):
        self.characters = list("abcdefghijklmnopqrstuvwxyz ")
        self._vocab_str_to_int = {
            "[CLS]": 0,
            "[SEP]": 1,
            "[BOS]": 2,
            "[EOS]": 3,
            "[MASK]": 4,
            "[PAD]": 5,
            "[RESERVED]": 6,
            "[UNK]": 7,
            **{ch: i + 8 for i, ch in enumerate(self.characters)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str, **kwargs) -> typing.List[str]:
        return list(text.lower())

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def get_vocab(self) -> typing.Dict[str, int]:
        return self._vocab_str_to_int


class Alpha8Tokenizer(transformers.PreTrainedTokenizer):
    """Minimal tokenizer for 26 lowercase letters.

    IDs:
      - 'a'..'z' -> 0..25
      - [BOS]=26, [EOS]=27, [PAD]=28, [UNK]=29
    """

    def __init__(
        self,
        bos_token="[BOS]",
        eos_token="[EOS]",
        pad_token="[PAD]",
        unk_token="[UNK]",
        sep_token=None,
        cls_token=None,
        mask_token=None,
        **kwargs,
    ):
        self.characters = list("abcdefghijklmnopqrstuvwxyz")
        self._vocab_str_to_int = {
            "[BOS]": 26,
            "[EOS]": 27,
            "[PAD]": 28,
            "[UNK]": 29,
            **{ch: i for i, ch in enumerate(self.characters)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str, **kwargs) -> typing.List[str]:
        return list(text.lower())

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def get_vocab(self) -> typing.Dict[str, int]:
        return self._vocab_str_to_int


def build_tokenizer(name_or_path, vocab_size=None):
    """Build the tokenizer named by `data.tokenizer_name_or_path`.

    `vocab_size` is only consulted by the 'synthetic-align' tokenizer.
    """
    if name_or_path == "text8":
        tokenizer = Text8Tokenizer()
    elif name_or_path == "bert-base-uncased":
        tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    elif name_or_path == "synthetic-align":
        tokenizer = SyntheticTokenizer(vocab_size=vocab_size + 2)
    elif name_or_path in ["alpha8", "synthetic-alpha8"]:
        tokenizer = Alpha8Tokenizer()
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(name_or_path)

    # This post-processor is why `datasets.bos_eos_ids` reads ids back through
    # encode() instead of using bos_token_id/eos_token_id.
    if isinstance(tokenizer, transformers.GPT2TokenizerFast) or isinstance(
        tokenizer, transformers.GPT2Tokenizer
    ):
        tokenizer._tokenizer.post_processor = tokenizers.processors.BertProcessing(
            (tokenizer.bos_token, tokenizer.bos_token_id),
            (tokenizer.eos_token, tokenizer.eos_token_id),
        )

    # For wrapped batches:
    #  [BOS] sent1 [EOS] sent2-fragment [EOS]
    #  [BOS] sent2-fragment [EOS] sent3 [EOS]
    if tokenizer.bos_token is None:
        if tokenizer.cls_token is None:
            raise AttributeError(
                f"Tokenizer must have a bos_token or cls_token: {tokenizer}"
            )
        tokenizer.bos_token = tokenizer.cls_token
    if tokenizer.eos_token is None:
        if tokenizer.sep_token is None:
            raise AttributeError(
                f"Tokenizer must have a eos_token or sep_token: {tokenizer}"
            )
        tokenizer.eos_token = tokenizer.sep_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    return tokenizer


def get_tokenizer(config):
    return build_tokenizer(
        config.data.tokenizer_name_or_path,
        vocab_size=config.data.get("vocab_size", None),
    )
