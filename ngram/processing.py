import re
from typing import Iterable, List, Union, Optional
import string
from pathlib import Path
from tqdm import tqdm
from ngram.abstract import Object


class Processing(Object):
    pass


SAFE = string.ascii_letters + string.digits + "!',-.:;?`\" "
SURROGATES = r"[\ud800-\udfff]"  # for when what's read in is not utf-8 compatible
UNSAFE = rf'[{re.escape("".join(set(string.printable) - set(SAFE)))}]|{SURROGATES}'
LINE_ENDING = r"[.!?]"
LINE_SEPARATION = r"[;:-]"
ALL_PUNC = rf"[{re.escape(string.punctuation)}]"
CONTRACTIONS = r"n\'?t|\'?s|\'?m|\'?re|\'?ve|\'?ll|\'?d"


def augment_lines(lines: Iterable[str], disable_tqdm: bool = False) -> Iterable[str]:
    for line in tqdm(lines, desc="Augmenting lines", disable=disable_tqdm):
        # remove cornell movie dialogues metadata
        if "+++$+++" in line:
            line = line.split("+++$+++")[-1]
        # replace unicode quotes with ascii quotes (i think this might not actually occur anymore)
        line = re.sub(r"‘|’", "'", line)
        line = re.sub(r"“|”", '"', line)
        # remove hyphens (in words like "well-known")
        line = re.sub(r"(\w+)-(\w+)", r"\1\2", line)
        # remove anything beginning with "@" (metadata)
        line = re.sub(r"@\S*", " - ", line)  # "-" so it gets line-broken later
        # remove text in square brackets or parentheses
        line = re.sub(r"\[.*?\]", "", line)
        line = re.sub(r"\(.*?\)", "", line)
        # remove double-quotes
        line = re.sub(r'"', "", line)
        # turn ".!" into "!"
        line = re.sub(r"\.!", r"!", line)
        # remove space between "7: 30" and so on
        line = re.sub(r"(\d+):\s+(\d+)", r"\1:\2", line)
        # remove space before n't, 'll and so on
        line = re.sub(rf"([a-z])\s+({CONTRACTIONS})\b", r"\1\2", line, re.IGNORECASE)
        # remove the space between gon na, wan na
        line = re.sub(r"(gon|wan)\s+na\b", r"\1na", line, re.IGNORECASE)
        # remove spaces before punctuation
        line = re.sub(rf"\s+({ALL_PUNC})", r"\1", line)
        # replace multiple spaces with a single space
        line = re.sub(r"\s+", " ", line)
        yield line


def split_lines(lines: Iterable[str], disable_tqdm: bool = False) -> Iterable[str]:
    for line in tqdm(lines, desc="Splitting lines", disable=disable_tqdm):
        for split_line in re.split(rf"(.*?(?:{LINE_ENDING}|{LINE_SEPARATION}))", line):
            yield split_line


def filter_and_fix_lines(
    lines: Iterable[str], disable_tqdm: bool = False
) -> Iterable[str]:
    for line in tqdm(lines, desc="Filtering and fixing lines", disable=disable_tqdm):
        line = line.strip()
        if not line:
            continue

        # must start with a letter
        if not re.match(r"[a-zA-Z]", line):
            continue
        # must not contain certain characters
        if re.search(UNSAFE, line):
            continue
        # must not have 4+ consecutive punctuation
        if re.search(rf"{ALL_PUNC}{{4,}}", line):
            continue
        # remove 'voice-over' from line (cornell movies)
        line = re.sub(r"Voice-over", "", line)
        # remove any lingering separation symbols
        line = re.sub(LINE_SEPARATION, "", line)
        # add period to end of line if no punctuation
        if not re.match(LINE_ENDING, line[-1]):
            line += "."
        # some weird cases (just bad data) where a contraction starts the line
        if re.match(rf"({CONTRACTIONS})\b", line, re.IGNORECASE):
            continue
        # capture any missing cases (not totally sure why they exist)
        line = re.sub(rf"([a-z])\s+({CONTRACTIONS})\b", r"\1\2", line, re.IGNORECASE)
        # must be at least 5 words and no more than 50
        if not 5 <= len(re.findall(r"\w+(?:'\w+)?", line)) <= 50:
            continue

        line = line.strip()
        yield line


def process_text(text: str) -> str:
    # lowercase
    text = text.lower()
    # remove hyphens (in words like "well-known")
    text = re.sub(r"(\w+)-(\w+)", r"\1\2", text)
    # strip punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)
    # remove space before n't, 'll and so on
    text = re.sub(rf"([a-z])\s+({CONTRACTIONS})\b", r"\1\2", text)
    # remove the space between gon na, wan na
    text = re.sub(r"(gon|wan)\s+na\b", r"\1na", text)
    return text


def tokenize(text: str) -> List[str]:
    return process_text(text).split()


def get_lines(file: Union[str, Path], disable_tqdm: bool = False) -> Iterable[str]:
    with open(file, "rt", encoding="utf-8", errors="surrogateescape") as f:
        for line in tqdm(f, desc="Getting lines", disable=disable_tqdm):
            yield line


def process_file(file: Union[str, Path], disable_tqdm: bool = False) -> Iterable[str]:
    lines = get_lines(file, disable_tqdm=True)
    lines = augment_lines(lines, disable_tqdm=True)
    lines = split_lines(lines, disable_tqdm=True)
    lines = filter_and_fix_lines(lines, disable_tqdm=True)
    # ^stop here for simply cleaned data (capitalization and some punctuation kept)
    for line in tqdm(lines, desc="Processing text", disable=disable_tqdm):
        # final processing for ngram model
        yield process_text(line)


def preprocess_files(
    input_folder: Union[str, Path],
    output_folder: Union[str, Path],
    combine_files_as: Optional[Union[str, Path]] = None,
    disable_tqdm: bool = False,
) -> Path:
    files = list(Path(input_folder).rglob("*.txt"))
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    if combine_files_as is not None:
        output_file = output_folder / combine_files_as
        with open(output_file, "wt", encoding="utf-8") as f:
            for file in tqdm(files, desc="Preprocessing files", disable=disable_tqdm):
                for line in process_file(file, disable_tqdm=True):
                    f.write(line + "\n")
    else:
        for file in tqdm(files, desc="Preprocessing files", disable=disable_tqdm):
            output_file = output_folder / file.relative_to(input_folder)
            with open(output_file, "wt", encoding="utf-8") as f:
                for line in process_file(file, disable_tqdm=True):
                    f.write(line + "\n")
