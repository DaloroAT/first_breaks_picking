.PHONY: install_precommit
install_precommit:
	python -m pip install --upgrade pre-commit==2.15.0

.PHONY: run_precommit
run_precommit:
	pre-commit install && pre-commit run -a


.PHONY: tests
tests:
	pytest -sv --disable-warnings tests


.PHONY: build_wheel
build_wheel:
	python -m pip install --upgrade pip
	python3 -m pip install --upgrade twine
	pip install --upgrade pip setuptools wheel
	rm -rf dist build first_breaks_picking.egg-info
	python3 setup.py sdist bdist_wheel

.PHONY: upload_to_pip
upload_to_pip: build_wheel
	twine upload dist/*
