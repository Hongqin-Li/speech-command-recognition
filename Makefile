ALL:

.PHONY: install lint test

install:
	pip install -r scripts/requirements.txt
	pip install flake8
	pip install pytest

lint:
	flake8 . --count --show-source --statistics --exclude=raw_datasets,datasets

test:
	pytest --ignore=raw_datasets --ignore=datasets
