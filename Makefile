POETRY ?= poetry
PYTHON_PATHS := \
	src/drl_navigation_ros2 \
	src/v550_ackermann_simulations/v550_ackermann_gazebo/launch \
	src/v550_ackermann_simulations/v550_ackermann_gazebo/scripts \
	tests

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  install  Install runtime, test, and lint dependencies"
	@echo "  format   Apply Ruff fixes and formatting"
	@echo "  lint     Check Python source without modifying files"
	@echo "  test     Run the Python test suite"

.PHONY: install
install:
	@command -v $(POETRY) >/dev/null || { echo "Poetry is required: https://python-poetry.org/docs/"; exit 2; }
	$(POETRY) install --with dev,tests,linters

.PHONY: format
format:
	$(POETRY) run ruff check --fix $(PYTHON_PATHS)
	$(POETRY) run ruff format $(PYTHON_PATHS)

.PHONY: lint
lint:
	$(POETRY) run ruff check $(PYTHON_PATHS)
	$(POETRY) run ruff format --check $(PYTHON_PATHS)

.PHONY: test
test:
	$(POETRY) run pytest
