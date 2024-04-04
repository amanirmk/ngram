# ngram

### 1. Download and setup module
```
git clone https://github.com/amanirmk/ngram.git
cd ngram
make env
```
This should work for most devices, but I encourage inspecting the Makefile before running any `make` commands if you are concerned.

### 2. Download and setup KenLM
```
make kenlm
```
Unfortunately this command will only work on some devices (namely my 2021 Macbook running Apple M1 Pro). You should probably setup KenLM on your own and specify the correct path in `ngram/args.py`. See [KenLM](https://kheafield.com/code/kenlm/) for more.

### 3. Use the module (two options)

To run exactly as pre-specified in `ngram/args.py`:
```
make ngram
```
To override defaults with command line arguments:
```
python -m ngram [args]
```

_If you are contributing_, please run `make format` and `make test` before submitting a pull request.
