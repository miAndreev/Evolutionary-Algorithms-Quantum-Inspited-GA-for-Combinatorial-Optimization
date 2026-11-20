makefiles = Makefile
venv_path = .
python_interpeter = "python3.14"

# code checks
.PHONY: init_venv # Init test/dev venv
init_venv:
	$(eval venv_path=${venv_path}/dev_venv)
	${python_interpeter} -m venv ${venv_path}
	${venv_path}/bin/pip install -U pip uv
	VIRTUAL_ENV=${venv_path} ${venv_path}/bin/uv pip install --no-cache -e .

.PHONY: init_lint # Init linters
init_lint: init_venv
	VIRTUAL_ENV=${venv_path} ${venv_path}/bin/uv pip install --no-cache -U -r requirements/linters.txt

.PHONY: format # Format /src and /tests
format: init_lint
	${venv_path}/bin/black ./src

.PHONY: check # Check /src with linters
check: init_lint
	${venv_path}/bin/flake8 ./src || true
	${venv_path}/bin/pylint ./src || true
	${venv_path}/bin/bandit -r ./src || true
	${venv_path}/bin/iscan ./src || true
	${venv_path}/bin/ruff check ./src || true

.PHONY: check_tests # Check /tests with linters
check_tests: init_lint
	${venv_path}/bin/flake8 ./tests || true
	${venv_path}/bin/pylint ./tests || true

# test code
.PHONY: test_init # Setup the dev_venv for tests
test_init: init_venv
	VIRTUAL_ENV=${venv_path} ${venv_path}/bin/uv pip install --no-cache -U -r requirements/tests.txt

.PHONY: test # Run tests
test: test_init
	umask 0022
	${venv_path}/bin/pytest -v

.PHONY: cover # Run code coverage
cover: test_init
	umask 0022
	${venv_path}/bin/pytest -vv -- tests

# experiments
SPEC ?= experiments/smoke.yaml
WORKERS ?= 4
RESULTS ?= results/smoke

.PHONY: experiment # Run an experiment: make experiment SPEC=experiments/smoke.yaml WORKERS=4
experiment: init_venv
	${venv_path}/bin/python -m qiga run $(SPEC) -w $(WORKERS)

.PHONY: experiments_all # Run the full study: comparison, repair and both QIGA sweeps
experiments_all: init_venv
	@for spec in comparison repair sweep_knapsack sweep_tsp; do \
		${venv_path}/bin/python -m qiga run experiments/$$spec.yaml -w $(WORKERS) || exit 1; \
	done

.PHONY: report # Write figures, summary and statistical tests for the latest run: make report RESULTS=results/smoke
report: init_venv
	VIRTUAL_ENV=${venv_path} ${venv_path}/bin/uv pip install --no-cache -r requirements/viz.txt
	${venv_path}/bin/python -m qiga summary $(RESULTS)
	${venv_path}/bin/python -m qiga_viz report $(RESULTS)

.PHONY: reports_all # Write reports for every experiment of the full study
reports_all: init_venv
	VIRTUAL_ENV=${venv_path} ${venv_path}/bin/uv pip install --no-cache -r requirements/viz.txt
	@for spec in comparison repair sweep_knapsack sweep_tsp; do \
		${venv_path}/bin/python -m qiga_viz report results/$$spec || exit 1; \
	done

.PHONY: stats # Significance tests for the latest run: make stats RESULTS=results/comparison METRIC=anytime
METRIC ?= gap
stats: init_venv
	${venv_path}/bin/python -m qiga stats $(RESULTS) -m $(METRIC)

.PHONY: demo # Trace one QIGA run on TSP: Q-bit heatmap, best tour and tour animation in results/demo
demo: init_venv
	VIRTUAL_ENV=${venv_path} ${venv_path}/bin/uv pip install --no-cache -r requirements/viz.txt
	${venv_path}/bin/python -m qiga_viz demo tsp -n 30 -e 10000

.PHONY: dist_clean # Clean all generated files
dist_clean: clean_venv
	rm -rf htmlcov
	rm -f .coverage
	rm -rf .pytest_cache
	rm -rf dist
	rm -rf build
	rm -rf *.pex
	find . -type d -iname __pycache__ -prune -exec rm -rf {} \;

.PHONY: clean # Clean the repo
clean: clean_venv

.PHONY: clean_venv # Remove the venvs
clean_venv:
	rm -rf ${venv_path}/dev_venv
	rm -rf ${venv_path}/build_venv
	rm -rf ${venv_path}/copier_venv

.PHONY: update_deps # Update python dependencies versions
update_deps: init_venv
	@for file in requirements/*.in; do \
		VIRTUAL_ENV=${venv_path} ${venv_path}/bin/uv pip compile $$file -o $${file%.in}.txt -n -U; \
	done

.PHONY: pre-commit_install # Install the pre-commit hooks
pre-commit_install:
	pre-commit install -t pre-commit -t prepare-commit-msg

.PHONY: pre-commit_runall # Run pre-commit for all files
pre-commit_runall:
	pre-commit run --all-files

.PHONY: help # Generate list of targets with descriptions
help:
	@grep -h '^.PHONY: .* #' $(makefiles) | sed 's/\.PHONY: \(.*\) # \(.*\)/\1 \2/' | expand -t20
