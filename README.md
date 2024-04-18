# ngram

### 1. Download and setup module
```
git clone https://github.com/amanirmk/ngram.git
cd ngram
make env
```
This should work for most devices, but I encourage inspecting the Makefile before running any `make` commands if you are concerned.

### 2. Use the module (two options)

To run exactly as pre-specified in `ngram/args.py`:
```
make ngram
```
To override defaults with command line arguments:
```
python -m ngram [args]
```

_If you are contributing_, please run `make format` and `make test` before submitting a pull request.
