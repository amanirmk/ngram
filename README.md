# ngram

```
# 1. Setup environment
make env
# 2. Setup kenlm
# This command will only work on some devices (namely my 2021 Macbook running Apple M1 Pro)
# You should probably setup kenlm on your own and specify the correct path in ngram/args.py
make kenlm
# 3. Use the module (two options)
make ngram # run exactly as specified in ngram/args.py
python -m ngram # can override defaults with cmd line arguments
```

If contributing, run `make format` and `make test` before submitting a pull request.