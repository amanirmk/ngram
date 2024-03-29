import re
import typing
import string
import os
from tqdm import tqdm
from pathlib import Path
from ngram.datatypes import NGram
from ngram.model import Model

SAFE = string.ascii_letters + string.digits + "!',-.:;?`\" "
UNSAFE = rf'[{re.escape("".join(set(string.printable) - set(SAFE)))}]'
LINE_ENDING = r"[.!?]"
LINE_SEPARATION = r"[;:-]"
ALL_PUNC = rf"[{re.escape(string.punctuation)}]"
CONTRACTIONS = r"n\'?t|\'?s|\'?m|\'?re|\'?ve|\'?ll|\'?d"


def augment_text(text: str):
    # remove anything beginning with "@"
    text = re.sub(r"@\S*", "-", text)  # <-- so it gets line-broken later
    # remove text in square brackets
    text = re.sub(r"\[.*?\]", "", text)
    # remove double-quotes
    text = re.sub(r'"', "", text)
    # remove hyphens (in words like "well-known")
    text = re.sub(r"(\w+)-(\w+)", r"\1\2", text)
    # turn ".!" into "!"
    text = re.sub(r"\.!", r"!", text)
    # remove space between "7: 30" and so on
    text = re.sub(r"(\d+):\s+(\d+)", r"\1:\2", text)
    # remove spaces before punctuation
    text = re.sub(rf"\s+({ALL_PUNC})", r"\1", text)
    # remove space before n't, 'll and so on
    text = re.sub(rf"([a-zA-Z])\s+({CONTRACTIONS})\b", r"\1\2", text, re.IGNORECASE)
    # remove the space between gon na, wan na
    text = re.sub(r"(gon|wan)\s+na\b", r"\1na", text, re.IGNORECASE)
    # replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)
    return text


def split_lines(lines: typing.Iterable[str]) -> typing.Iterable[str]:
    for line in tqdm(lines, desc="Splitting lines"):
        for split_line in re.split(rf"(.*?(?:{LINE_ENDING}|{LINE_SEPARATION})) ", line):
            yield split_line


def filter_and_fix_lines(lines: typing.Iterable[str]) -> typing.Iterable[str]:
    for line in tqdm(lines, desc="Filtering and fixing lines"):
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
        # remove all lingering separation symbols
        line = re.sub(LINE_SEPARATION, "", line)
        # add period to end of line if no punctuation
        if not re.match(LINE_ENDING, line[-1]):
            line += "."
        # must be at least 5 words and no more than 50
        if not 5 <= len(re.findall(r"\w+", line)) <= 50:
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
    return text


def tokenize(text: typing.List[str]) -> typing.List[str]:
    return process_text(text).split()


def process_file(file: typing.Union[str, Path]) -> typing.Iterable[str]:
    with open(file, "rt", encoding="utf-8") as f:
        lines = []
        for line in tqdm(f.readlines(), desc="Reading and augmenting lines"):
            lines.append(augment_text(line))

    lines = split_lines(lines)
    lines = filter_and_fix_lines(lines)
    for line in tqdm(lines, desc="Processing text"):
        yield process_text(line)


def preprocess_files(
    input_folder: typing.Union[str, Path], output_file: typing.Union[str, Path]
) -> Path:
    files = list(Path(input_folder).rglob("*.txt"))
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    for file in tqdm(files, desc="Preprocessing files"):
        with open(output_file, "at", encoding="utf-8") as f:
            for line in process_file(file):
                f.write(line + "\n")
    return output_file


def create_arpa(
    text_file: typing.Union[str, Path], arpa_path: typing.Union[str, Path], n
) -> None:
    flag = os.system(f"./kenlm/build/bin/lmplz -o {n} <{text_file} >{arpa_path}")
    if flag != 0:
        raise ValueError("Error in creating ARPA file")


def create_binary(
    arpa_path: typing.Union[str, Path], binary_path: typing.Union[str, Path]
) -> None:
    flag = os.system(f"./kenlm/build/bin/build_binary {arpa_path} {binary_path}")
    if flag != 0:
        raise ValueError("Error in creating binary file")


def create_arpa_and_binary(
    text_file: typing.Union[str, Path],
    output_folder: typing.Union[str, Path],
    n: int,
    all_up_to: bool = False,
) -> None:
    text_file = Path(text_file)
    arpa_folder = Path(output_folder) / "arpa"
    binary_folder = Path(output_folder) / "bin"
    arpa_folder.mkdir(parents=True, exist_ok=True)
    binary_folder.mkdir(parents=True, exist_ok=True)

    if all_up_to:
        for k in range(2, n + 1):
            arpa_path = arpa_folder / Path(text_file.stem + f"_{k}.arpa")
            binary_path = binary_folder / Path(text_file.stem + f"_{k}.binary")
            create_arpa(text_file, arpa_path, k)
            create_binary(arpa_path, binary_path)
    else:
        arpa_path = arpa_folder / Path(text_file.stem + f"_{n}.arpa")
        binary_path = binary_folder / Path(text_file.stem + f"_{n}.binary")
        create_arpa(text_file, arpa_path, n)
        create_binary(arpa_path, binary_path)


def create_ngram(
    arpa: typing.Union[str, Path],
    binary: typing.Union[str, Path],
    output_file: typing.Union[str, Path],
) -> None:
    arpa_path = Path(arpa)
    binary_path = Path(binary)
    ngram_path = Path(output_file)
    ngram_path.parent.mkdir(parents=True, exist_ok=True)

    t = tqdm(desc="Reading ngrams")
    with open(arpa_path, "rt", encoding="utf-8") as f:
        f.readline()
        line = f.readline()
        # find order of the model
        while "ngram" in line:
            n = int(line.split()[1][0])
            line = f.readline()

        # find start of the ngrams
        while True:
            if f"{n}-grams" in f.readline():
                break

        # read the ngrams
        ngrams = []
        while True:
            split_line = f.readline().split()
            if not split_line:
                break
            _, *tokens = split_line
            ngrams.append(NGram(tokens=tokens))
            t.update()
    t.close()

    m = Model(binary_path)
    scored_ngrams = [
        (ngram, m.freq_per_mil(ngram)[0])
        for ngram in tqdm(ngrams, desc="Getting ngram frequencies")
    ]
    print("Sorting ngrams")
    scored_ngrams.sort(key=lambda x: -x[1])

    with open(ngram_path, "wt", encoding="utf-8") as f:
        for ngram, fpm in tqdm(scored_ngrams, desc="Writing ngrams"):
            f.write(f"{fpm} {ngram.text()}\n")


def create_model_files(
    input_folder: typing.Union[str, Path],
    processed_corpora_folder: typing.Union[str, Path],
    model_output_folder: typing.Union[str, Path],
    n,
    filestem: str = "all_corpora",
    all_up_to=True,
) -> None:
    input_folder = Path(input_folder)
    text_file = Path(processed_corpora_folder) / Path(filestem + ".txt")
    arpa_file = Path(model_output_folder) / "arpa" / Path(filestem + ".arpa")
    binary_file = Path(model_output_folder) / "bin" / Path(filestem + ".binary")
    ngram_file = Path(model_output_folder) / "ngram" / Path(filestem + ".ngram")

    preprocess_files(input_folder, text_file)
    create_arpa_and_binary(text_file, model_output_folder, n, all_up_to)
    if all_up_to:
        for k in range(2, n + 1):
            create_ngram(
                arpa_file,
                binary_file,
                ngram_file.parent / Path(ngram_file.stem + f"_{k}.ngram"),
            )
    else:
        create_ngram(
            arpa_file,
            binary_file,
            ngram_file.parent / Path(ngram_file.stem + f"_{n}.ngram"),
        )
