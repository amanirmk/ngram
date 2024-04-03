# ngram

1. Download and setup module
```
git clone https://github.com/amanirmk/ngram.git
make env
```
2. Download and setup KenLM

Unfortunately this command will only work on some devices (namely my 2021 Macbook running Apple M1 Pro). You should probably setup KenLM on your own and specify the correct path in `ngram/args.py`. See [KenLM](https://kheafield.com/code/kenlm/) for more.
```
make kenlm
```
3. Use the module (two options)

To run exactly as pre-specified in `ngram/args.py`:
```
make ngram
```
To override defaults with command line arguments:
```
python -m ngram [args]
```

If you are contributing, please run `make format` and `make test` before submitting a pull request.
