.PHONY: run_precommit
run_precommit:
	pre-commit run -a

.PHONY: run_tests
run_tests:
	pytest -sv --disable-warnings tests

.PHONY: build_wheel
build_wheel:
	python -m pip install --upgrade pip
	python -m pip install --upgrade twine
	pip install --upgrade pip setuptools wheel
	rm -rf dist build first_breaks_picking.egg-info
	python setup.py sdist bdist_wheel

.PHONY: upload_to_pip
upload_to_pip: build_wheel
	twine upload dist/*
