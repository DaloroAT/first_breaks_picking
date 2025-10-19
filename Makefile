.PHONY: run_precommit
run_precommit:
	pre-commit run -a

.PHONY: run_tests
run_tests:
	pytest -sv --disable-warnings tests

