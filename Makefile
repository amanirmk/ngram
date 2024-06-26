SHELL := /usr/bin/env bash
EXEC := python=3.10
PACKAGE := ngram
RUN := python -m
INSTALL := $(RUN) pip install
ACTIVATE := source activate $(PACKAGE)
.DEFAULT_GOAL := help

## help      : print available build commands.
.PHONY : help
help : Makefile
	@sed -n 's/^##//p' $<

## env       : setup environment and install dependencies.
.PHONY : env
env : $(PACKAGE).egg-info/
$(PACKAGE).egg-info/ : setup.py requirements.txt
ifeq (0, $(shell conda env list | grep -wc $(PACKAGE)))
	@conda create -yn $(PACKAGE) $(EXEC)
endif
	@$(ACTIVATE); $(INSTALL) -e "."

## format    : format code with black.
.PHONY : format
format : env
	@black .

## test      : run testing pipeline.
.PHONY : test
test: style static
style : black
static : mypy pylint 
mypy : env
	@mypy \
	-p $(PACKAGE) \
	--ignore-missing-imports
pylint : env
	@pylint $(PACKAGE) \
	--disable C0112,C0113,C0114,C0115,C0116,R1710,R0903,R0913,R0914,C0200,R0902 \
	|| pylint-exit $$?
black : env
	@black --check .

## ngram     : run package.
.PHONY : $(PACKAGE)
$(PACKAGE) : env
	@$(RUN) $(PACKAGE)

## uninstall : remove environment
.PHONY : uninstall
uninstall :
	@conda env remove -yn $(PACKAGE); touch requirements.txt
