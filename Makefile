.PHONY: ci
ci: lint tests wheel egg

.PHONY: lint
lint:
	flake8 allrank
	flake8 tests
	mypy allrank --ignore-missing-imports  --check-untyped-defs
	mypy tests --ignore-missing-imports --check-untyped-defs

.PHONY: install-reqs
install-reqs:
	pip install -r requirements.txt
	python setup.py install

.PHONY: tests
tests: install-reqs unittests

.PHONY: unittests
unittests:
	python -m pytest
