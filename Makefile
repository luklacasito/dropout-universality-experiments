PYTHON ?= python

.PHONY: check lint format test check-shell figures

check: lint check-shell test

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

test:
	$(PYTHON) -m pytest -q

figures:
	MPLBACKEND=Agg $(PYTHON) experiments/paper/make_appendix.py

check-shell:
	@find experiments -type f \( -name '*.sh' -o -name '*.sbatch' \) -exec bash -c 'for script do bash -n "$$script" || exit; done' _ {} +
