import re
import typing
import string
import os
from tqdm import tqdm
from pathlib import Path
from ngram.datatypes import NGram
from ngram.model import Model
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


def augment_lines(
    lines: typing.Iterable[str], disable_tqdm: bool = False
) -> typing.Iterable[str]:
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


def split_lines(
    lines: typing.Iterable[str], disable_tqdm: bool = False
) -> typing.Iterable[str]:
    for line in tqdm(lines, desc="Splitting lines", disable=disable_tqdm):
        for split_line in re.split(rf"(.*?(?:{LINE_ENDING}|{LINE_SEPARATION}))", line):
            yield split_line


def filter_and_fix_lines(
    lines: typing.Iterable[str], disable_tqdm: bool = False
) -> typing.Iterable[str]:
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


def tokenize(text: str) -> typing.List[str]:
    return process_text(text).split()


def get_lines(
    file: typing.Union[str, Path], disable_tqdm: bool = False
) -> typing.Iterable[str]:
    with open(file, "rt", encoding="utf-8", errors="surrogateescape") as f:
        for line in tqdm(f, desc="Getting lines", disable=disable_tqdm):
            yield line


def process_file(
    file: typing.Union[str, Path], disable_tqdm: bool = False
) -> typing.Iterable[str]:
    lines = get_lines(file, disable_tqdm=True)
    lines = augment_lines(lines, disable_tqdm=True)
    lines = split_lines(lines, disable_tqdm=True)
    lines = filter_and_fix_lines(lines, disable_tqdm=True)
    # ^stop here for simply cleaned data (capitalization and some punctuation kept)
    for line in tqdm(lines, desc="Processing text", disable=disable_tqdm):
        # final processing for ngram model
        yield process_text(line)


def preprocess_files(
    input_folder: typing.Union[str, Path],
    output_file: typing.Union[str, Path],
    disable_tqdm: bool = False,
) -> Path:
    files = list(Path(input_folder).rglob("*.txt"))
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "at", encoding="utf-8") as f:
        for file in tqdm(files, desc="Preprocessing files", disable=disable_tqdm):
            for line in process_file(file, disable_tqdm=True):
                f.write(line + "\n")
    return output_file


def create_arpa(
    text_file: typing.Union[str, Path],
    arpa_path: typing.Union[str, Path],
    n,
    kenlm_bin_path: typing.Union[str, Path],
    kenlm_tmp_path: typing.Optional[typing.Union[str, Path]] = None,
    kenlm_ram_limit_mb: typing.Optional[int] = None,
    prune: bool = True,
) -> None:
    cmd_path = Path(kenlm_bin_path) / "lmplz"
    cmd_str = f"{cmd_path} -o {n}"
    if kenlm_tmp_path is not None:
        cmd_str += f" -T {kenlm_tmp_path}"
    if kenlm_ram_limit_mb is not None:
        cmd_str += f" -S {kenlm_ram_limit_mb}M"
    if prune:
        cmd_str += " --prune 0 1"
    cmd_str += f" <{text_file} >{arpa_path}"
    flag = os.system(cmd_str)
    if flag != 0:
        raise ValueError(f"Error in creating ARPA file (cmd: {cmd_str})")


def create_binary(
    arpa_path: typing.Union[str, Path],
    binary_path: typing.Union[str, Path],
    kenlm_bin_path: typing.Union[str, Path],
) -> None:
    cmd_path = Path(kenlm_bin_path) / "build_binary"
    cmd_str = f"{cmd_path} {arpa_path} {binary_path}"
    flag = os.system(cmd_str)
    if flag != 0:
        raise ValueError(f"Error in creating binary file (cmd: {cmd_str})")


def create_arpa_and_binary(
    text_file: typing.Union[str, Path],
    model_output_folder: typing.Union[str, Path],
    n: int,
    kenlm_bin_path: typing.Union[str, Path],
    kenlm_tmp_path: typing.Union[str, Path],
    kenlm_ram_limit_mb: typing.Optional[int] = None,
    all_up_to: bool = False,
    prune: bool = True,
) -> None:
    text_file = Path(text_file)
    arpa_folder = Path(model_output_folder) / "arpa"
    binary_folder = Path(model_output_folder) / "bin"
    arpa_folder.mkdir(parents=True, exist_ok=True)
    binary_folder.mkdir(parents=True, exist_ok=True)

    if all_up_to:
        for k in range(2, n + 1):
            arpa_path = arpa_folder / Path(text_file.stem + f"_{k}.arpa")
            binary_path = binary_folder / Path(text_file.stem + f"_{k}.binary")
            create_arpa(
                text_file=text_file,
                arpa_path=arpa_path,
                n=k,
                kenlm_bin_path=kenlm_bin_path,
                kenlm_tmp_path=kenlm_tmp_path,
                kenlm_ram_limit_mb=kenlm_ram_limit_mb,
                prune=prune,
            )
            create_binary(
                arpa_path=arpa_path,
                binary_path=binary_path,
                kenlm_bin_path=kenlm_bin_path,
            )
    else:
        arpa_path = arpa_folder / Path(text_file.stem + f"_{n}.arpa")
        binary_path = binary_folder / Path(text_file.stem + f"_{n}.binary")
        create_arpa(
            text_file=text_file,
            arpa_path=arpa_path,
            n=n,
            kenlm_bin_path=kenlm_bin_path,
            kenlm_tmp_path=kenlm_tmp_path,
            kenlm_ram_limit_mb=kenlm_ram_limit_mb,
            prune=prune,
        )
        create_binary(
            arpa_path=arpa_path, binary_path=binary_path, kenlm_bin_path=kenlm_bin_path
        )


def read_ngrams(
    arpa: typing.Union[str, Path], get_unigram: bool = False
) -> typing.Iterable[NGram]:
    arpa_path = Path(arpa)
    with open(arpa_path, "rt", encoding="utf-8") as f:
        f.readline()
        line = f.readline()
        # find order of the model
        while "ngram" in line:
            n = int(line.split()[1][0])
            line = f.readline()

        if get_unigram:
            n = 1
        # find start of the ngrams
        while True:
            if f"{n}-grams" in f.readline():
                break

        # read the ngrams
        while True:
            split_line = f.readline().split()
            if not split_line:
                break
            _, *tokens = split_line
            if get_unigram:
                tokens = tokens[:1]
            yield NGram(tokens=tokens)


def read_ngram_file(
    ngram_file: typing.Union[str, Path],
    min_fpm: float = 0,
    only_freqs: bool = True,
    exclude_bos: bool = True,
    exclude_eos: bool = True,
    disable_tqdm: bool = False,
) -> typing.Union[typing.Iterable[float], typing.Iterable[typing.Tuple[float, NGram]]]:
    with open(ngram_file, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading frequencies", disable=disable_tqdm):
            freq, *tokens = line.split()
            if exclude_bos and tokens[0] == "<s>":
                continue
            if exclude_eos and tokens[-1] == "</s>":
                continue
            freq = float(freq)
            if freq < min_fpm:
                continue
            if only_freqs:
                yield freq
            yield freq, NGram(tokens=tokens)


def create_ngram(
    arpa: typing.Union[str, Path],
    binary: typing.Union[str, Path],
    output_file: typing.Union[str, Path],
    get_unigram: bool = False,
    disable_tqdm: bool = False,
) -> None:
    binary_path = Path(binary)
    ngram_path = Path(output_file)
    ngram_path.parent.mkdir(parents=True, exist_ok=True)

    m = Model(binary_path)
    ngrams = read_ngrams(arpa, get_unigram=get_unigram)
    scored_ngrams = [
        (ngram, m.freq_per_mil(ngram)[0])
        for ngram in tqdm(
            ngrams, desc="Getting ngram frequencies", disable=disable_tqdm
        )
    ]
    not disable_tqdm and print("Sorting ngrams")
    scored_ngrams.sort(key=lambda x: -x[1])

    with open(ngram_path, "wt", encoding="utf-8") as f:
        for ngram, fpm in tqdm(
            scored_ngrams, desc="Writing ngrams", disable=disable_tqdm
        ):
            f.write(f"{fpm} {ngram.text()}\n")


def process(
    input_folder: typing.Union[str, Path],
    processed_corpora_folder: typing.Union[str, Path],
    model_output_folder: typing.Union[str, Path],
    n,
    kenlm_bin_path: typing.Union[str, Path],
    kenlm_tmp_path: typing.Union[str, Path],
    kenlm_ram_limit_mb: typing.Optional[int] = None,
    proxy_n_for_unigram: typing.Optional[int] = None,
    filestem: str = "all_corpora",
    all_up_to: bool = True,
    prune: bool = True,
    disable_tqdm: bool = False,
) -> None:
    input_folder = Path(input_folder)
    text_file = Path(processed_corpora_folder) / Path(filestem + ".txt")
    arpa_file = Path(model_output_folder) / "arpa" / Path(filestem + ".arpa")
    binary_file = Path(model_output_folder) / "bin" / Path(filestem + ".binary")
    ngram_file = Path(model_output_folder) / "ngram" / Path(filestem + ".ngram")

    if n == 1:
        assert (
            proxy_n_for_unigram is not None
        ), "proxy_n_for_unigram must be provided if only creating unigram model"

    preprocess_files(input_folder, text_file, disable_tqdm=disable_tqdm)
    Processing.info(f"Preprocessed text saved to {text_file}.")
    create_arpa_and_binary(
        text_file=text_file,
        model_output_folder=model_output_folder,
        n=(proxy_n_for_unigram if n == 1 else n),
        kenlm_bin_path=kenlm_bin_path,
        kenlm_tmp_path=kenlm_tmp_path,
        kenlm_ram_limit_mb=kenlm_ram_limit_mb,
        all_up_to=all_up_to,
        prune=prune,
    )
    Processing.info(f"ARPA and binary files saved to {model_output_folder}.")
    if all_up_to:
        create_ngram(
            arpa_file.parent / Path(arpa_file.stem + f"_2.arpa"),
            binary_file.parent / Path(binary_file.stem + f"_2.binary"),
            ngram_file.parent / Path(ngram_file.stem + f"_1.ngram"),
            get_unigram=True,
            disable_tqdm=disable_tqdm,
        )
        for k in range(2, n + 1):
            create_ngram(
                arpa_file.parent / Path(arpa_file.stem + f"_{k}.arpa"),
                binary_file.parent / Path(binary_file.stem + f"_{k}.binary"),
                ngram_file.parent / Path(ngram_file.stem + f"_{k}.ngram"),
                disable_tqdm=disable_tqdm,
            )
    else:
        create_ngram(
            arpa_file.parent
            / Path(arpa_file.stem + f"_{proxy_n_for_unigram if n==1 else n}.arpa"),
            binary_file.parent
            / Path(binary_file.stem + f"_{proxy_n_for_unigram if n==1 else n}.binary"),
            ngram_file.parent / Path(ngram_file.stem + f"_{n}.ngram"),
            get_unigram=n == 1,
            disable_tqdm=disable_tqdm,
        )
    Processing.info(f"NGram files saved to {ngram_file.parent}.")
