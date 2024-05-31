IMAGE_NAME ?= first-breaks-picking:latest


.PHONY: install_precommit
install_precommit:
	python -m pip install --upgrade pre-commit==3.5.0

.PHONY: run_precommit
run_precommit:
	pre-commit install && pre-commit run -a

.PHONY: run_tests
run_tests:
	pytest -sv --disable-warnings tests


.PHONY: docker_build
docker_build:
	DOCKER_BUILDKIT=1 docker build -t $(IMAGE_NAME) .

.PHONY: docker_tests
docker_tests: docker_build
	docker run -t $(IMAGE_NAME) make run_tests


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
