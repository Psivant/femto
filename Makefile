PACKAGE_NAME := femto

CONDA_ENV_RUN=conda run --no-capture-output --name $(PACKAGE_NAME)

TEST_ARGS := -v --cov=$(PACKAGE_NAME) --cov-report=term --cov-report=xml --junitxml=unit.xml --color=yes

.PHONY: env lint format test docs docs-deploy docs-insiders

env:
	mamba create     --name $(PACKAGE_NAME)
	mamba env update --name $(PACKAGE_NAME) --file devtools/envs/base.yaml
	$(CONDA_ENV_RUN) pip install --no-deps -e .
	$(CONDA_ENV_RUN) pre-commit install || true

lint:
	$(CONDA_ENV_RUN) ruff check $(PACKAGE_NAME)
	$(CONDA_ENV_RUN) pre-commit run vulture --all-files || true

format:
	$(CONDA_ENV_RUN) ruff format $(PACKAGE_NAME)
	$(CONDA_ENV_RUN) ruff check --fix --select I $(PACKAGE_NAME)

test:
	$(CONDA_ENV_RUN) pytest -v $(TEST_ARGS) $(PACKAGE_NAME)/*/tests/

docs:
	$(CONDA_ENV_RUN) mkdocs build

docs-deploy:
ifndef VERSION
	$(error VERSION is not set)
endif
	$(CONDA_ENV_RUN) mike deploy --push --update-aliases $(VERSION)
