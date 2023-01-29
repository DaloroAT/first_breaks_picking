.PHONY: run_precommit
run_precommit:
	pre-commit install && pre-commit run -a
