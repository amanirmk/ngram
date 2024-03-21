from tqdm import tqdm


def arpa_to_ngram(arpa_path: str, n) -> None:
    t = tqdm(desc=f"Reading .arpa (n={n})")
    with open(arpa_path, "rt", encoding="utf-8") as f:
        # find start of the ngrams
        while True:
            if f"{n}-grams" in f.readline():
                break
            t.update()

        # read the ngrams
        ngrams = []
        while True:
            line = f.readline().split()
            if not line:
                break
            logprob, *tokens = line
            ngrams.append((float(logprob), tokens[:n]))
            t.update()
    t.close()

    # Sort the n-grams by logprob
    ngrams.sort(key=lambda x: -x[0])

    # Write the sorted n-grams to a file
    ngram_path = arpa_path.replace(".arpa", f"_{n}.ngram")
    with open(ngram_path, "wt", encoding="utf-8") as f:
        for ngram in tqdm(ngrams, desc=f"Writing .ngram (n={n})"):
            f.write(f"{ngram[0]} {' '.join(ngram[1])}\n")


def arpa_to_ngrams(arpa_path: str, min_n=None, max_n=None) -> None:
    min_n = min_n or 1
    if max_n is None:
        with open(arpa_path, "rt", encoding="utf-8") as f:
            f.readline()
            line = f.readline()
            while "ngram" in line:
                max_n = int(line.split()[1][0])
                line = f.readline()

    for n in range(min_n, max_n + 1):
        arpa_to_ngram(arpa_path, n)
