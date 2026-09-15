PYTHON ?= python

.PHONY: check lint format test test-rnn check-shell

check: lint check-shell test

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

test: test-rnn
	$(PYTHON) -m pytest -q

# Run each standalone study in a fresh process: each has its own run.py module.
test-rnn:
	$(PYTHON) -m unittest discover -s experiments/rnn_pilot -v
	$(PYTHON) -m unittest discover -s experiments/rnn_ptb -v
	$(PYTHON) -m unittest discover -s experiments/rnn_ptb_linear -v

check-shell:
	@find experiments -type f \( -name '*.sh' -o -name '*.sbatch' \) -exec bash -c 'for script do bash -n "$$script" || exit; done' _ {} +
