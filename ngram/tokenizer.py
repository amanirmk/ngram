import typing
import string
import re


class Tokenizer:
    EOS_SYMBOLS = "\.|!|\?"
    CONTRACTIONS = {
        "n't": "not",
        "'s": "is",
        "'m": "am",
        "'re": "are",
        "'ve": "have",
        "'ll": "will",
        "'d": "would",
    }
    SUBTOKEN_CHAR = "_"
    PUNC_TO_REMOVE = string.punctuation.replace("'", "")

    def __init__(
        self,
        lower: bool = True,
        remove_punc: bool = True,
        remove_meta: bool = True,
        split_contractions: bool = True,
        substitute_contractions: bool = False,
    ):
        self._lower = lower
        self._remove_punc = remove_punc
        self._remove_meta = remove_meta
        self._split_contractions = split_contractions
        self._substitute_contractions = substitute_contractions
        self._punc_table = str.maketrans("", "", self.PUNC_TO_REMOVE)

    def __call__(self, text: str) -> typing.List[str]:
        return self.tokenize_text(text)

    def tokenize_file(
        self,
        input_path: str,
        output_path: typing.Optional[str] = None,
        output_mode: str = "a",
        make_newlines: bool = False,
    ) -> typing.List[str]:
        assert output_mode in ["a", "w"], "Must append or write to output file"
        lines = []
        with open(input_path, "rt", encoding="utf-8") as f:
            if make_newlines:
                for line in f:
                    lines.extend(re.split(self.EOS_SYMBOLS, line))
            else:
                lines = f.readlines()
        for i in range(len(lines)):
            lines[i] = self.tokenize_text(lines[i])  # type: ignore[call-overload]
        if output_path is not None:
            with open(output_path, f"{output_mode}t", encoding="utf-8") as f:
                for line in lines:
                    if line:
                        f.write(" ".join(line) + "\n"*make_newlines)
        return lines

    def tokenize_text(self, text: str) -> typing.List[str]:
        words = text.split()
        tokens = []
        for word in words:
            tokens.extend(self._recursive_tokenize(word))
        return [t for t in tokens if t != ""]
    
    def process_text_for_kenlm(self, text: str) -> str:
        return " ".join(self.tokenize_text(text))
    
    def _split_by_contractions(self, word: str) -> typing.List[str]:
        make_subpattern = lambda c: fr"\S+(?={self.SUBTOKEN_CHAR}{c})|{self.SUBTOKEN_CHAR}{c}|\S+(?={c})|{c}"
        pattern = "|".join([make_subpattern(c) for c in self.CONTRACTIONS])
        pattern += "|'\S+|\S+"
        return re.findall(pattern, word, re.IGNORECASE | re.DOTALL)

    def _recursive_tokenize(self, word: str) -> typing.List[str]:
        if word == "":
            return []
        if self._remove_meta and word.startswith("@"):
            return []
        if self._split_contractions:
            # fix text that may have been split already
            if word in self.CONTRACTIONS.keys():
                word = self.SUBTOKEN_CHAR + word
            # split by contraction
            word = self._split_by_contractions(word)
            # recurse and append subtoken
            if len(word) > 1:
                return sum([self._recursive_tokenize(self.SUBTOKEN_CHAR*(i>0) + w) for i, w in enumerate(word)], [])
            else:
                word = word[0]
        if self._lower:
            word = word.lower()
        if self._substitute_contractions:
            if word[0] == self.SUBTOKEN_CHAR and word[1:] in self.CONTRACTIONS:
                word = self.CONTRACTIONS[word]
        if self._remove_punc:
            if word[0] == self.SUBTOKEN_CHAR:
                # don't remove subtoken character
                word = word[0] + word[1:].translate(self._punc_table)
            else:
                word = word.translate(self._punc_table)
        else:
            if word[-1] in string.punctuation:
                return [word[:-1], word[-1]]
        return [word]
