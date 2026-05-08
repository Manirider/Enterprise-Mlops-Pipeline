.PHONY: all setup install data pipeline test benchmark clean docker-build docker-up lint format help

PYTHON := python
PIP := pip
DVC := dvc
PYTEST := pytest

all: help

setup: install git-init dvc-init
	@echo "✓ Environment ready"

install:
	@echo "→ Installing Python dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✓ Dependencies installed"

git-init:
	@git init 2>/dev/null || true
	@echo "✓ Git initialized"

dvc-init:
	@$(DVC) init 2>/dev/null || true
	@echo "✓ DVC initialized"

data:
	@echo "→ Downloading UCI Adult Income dataset..."
	$(PYTHON) -c "\
import urllib.request, pathlib, sys; \
p = pathlib.Path('data'); p.mkdir(exist_ok=True); \
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'; \
print('Downloading from UCI ML Repository...'); \
urllib.request.urlretrieve(url, 'data/adult.csv'); \
print('Downloaded to data/adult.csv'); \
"
	@echo "✓ Dataset ready"

pipeline: dvc-init
	@echo "→ Running full DVC modular pipeline..."
	$(DVC) repro
	@echo "✓ Pipeline complete"

repro:
	$(DVC) repro

prepare:
	$(PYTHON) src/prepare.py

featurize:
	$(PYTHON) src/featurize.py

train:
	$(PYTHON) src/train.py

evaluate:
	$(PYTHON) src/evaluate.py

monolithic:
	@echo "→ Running monolithic ML pipeline (baseline)..."
	$(PYTHON) train_monolithic.py
	@echo "✓ Monolithic run complete"

exp-run:
	@echo "→ Running DVC experiment..."
	$(DVC) exp run

exp-show:
	$(DVC) exp show

exp-compare: exp-run exp-show

exp-n100:
	$(DVC) exp run --set-param train.n_estimators=100 --name exp-n100

exp-n200:
	$(DVC) exp run --set-param train.n_estimators=200 --name exp-n200

exp-depth5:
	$(DVC) exp run --set-param train.max_depth=5 --name exp-depth5

exp-depth15:
	$(DVC) exp run --set-param train.max_depth=15 --name exp-depth15

metrics:
	$(DVC) metrics show

metrics-diff:
	$(DVC) metrics diff

test:
	@echo "→ Running pytest test suite..."
	PYTHONPATH=. $(PYTEST) tests/ -v --tb=short
	@echo "✓ Tests complete"

test-cov:
	PYTHONPATH=. $(PYTEST) tests/ -v --cov=src --cov-report=term-missing --cov-report=html
	@echo "✓ Coverage report generated in htmlcov/"

test-fast:
	PYTHONPATH=. $(PYTEST) tests/ -x -q

benchmark:
	@echo "→ Running full benchmark comparison..."
	$(PYTHON) -c "\
import time, subprocess, sys; \
print('=== Monolithic Pipeline ==='); \
t0 = time.perf_counter(); \
subprocess.run([sys.executable, 'train_monolithic.py'], check=True); \
mono_time = time.perf_counter() - t0; \
print(f'Monolithic wall-clock: {mono_time:.2f}s'); \
print('=== DVC Pipeline (first run) ==='); \
t0 = time.perf_counter(); \
subprocess.run(['dvc', 'repro', '--force'], check=True); \
dvc_time = time.perf_counter() - t0; \
print(f'DVC first-run wall-clock: {dvc_time:.2f}s'); \
print('=== DVC Pipeline (cached re-run) ==='); \
t0 = time.perf_counter(); \
subprocess.run(['dvc', 'repro'], check=True); \
cache_time = time.perf_counter() - t0; \
print(f'DVC cached wall-clock: {cache_time:.2f}s'); \
print(f'Cache speedup: {mono_time/cache_time:.1f}x'); \
"
	@echo "✓ Benchmark complete"

test-cache:
	@echo "→ Validating DVC caching behavior..."
	bash test_caching.sh
	@echo "✓ Caching test complete. See repro_log.txt"

dag:
	@echo "→ Generating DVC DAG visualization..."
	$(DVC) dag
	@echo "(ASCII DAG shown above)"

docker-build:
	@echo "→ Building Docker image..."
	docker build -t enterprise-mlops-pipeline .
	@echo "✓ Image built: enterprise-mlops-pipeline"

docker-up:
	@echo "→ Starting Docker Compose services..."
	docker-compose up --build -d
	@echo "✓ Services started"

docker-down:
	docker-compose down

docker-test:
	docker-compose --profile test up --build

docker-pipeline:
	docker-compose run app dvc repro

clean-outputs:
	@echo "→ Removing generated ML artifacts..."
	rm -f data/processed.csv data/features.npz
	rm -f models/model.joblib models/model_monolithic.joblib
	rm -f metrics/scores.json metrics/scores_monolithic.json
	rm -f logs/*.log
	@echo "✓ Artifacts removed"

clean-cache:
	@echo "→ Clearing DVC cache..."
	$(DVC) cache clean
	@echo "✓ DVC cache cleared"

clean-pycache:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "✓ Python cache cleared"

clean: clean-outputs clean-pycache
	@echo "✓ Project cleaned"

clean-all: clean clean-cache
	@echo "✓ Full clean complete"

lint:
	ruff check src/ tests/ train_monolithic.py

format:
	black src/ tests/ train_monolithic.py

help:
	@echo ""
	@echo "Enterprise MLOps Pipeline — Makefile Commands"
	@echo "=============================================="
	@echo ""
	@echo "Setup:"
	@echo "  make setup          Install all dependencies + init git/dvc"
	@echo "  make data           Download UCI Adult dataset"
	@echo ""
	@echo "Pipeline:"
	@echo "  make pipeline       Run full DVC modular pipeline"
	@echo "  make monolithic     Run monolithic baseline script"
	@echo "  make prepare        Run Stage 1 only"
	@echo "  make featurize      Run Stage 2 only"
	@echo "  make train          Run Stage 3 only"
	@echo "  make evaluate       Run Stage 4 only"
	@echo ""
	@echo "Experiments:"
	@echo "  make exp-run        Run DVC experiment"
	@echo "  make exp-show       Show experiment comparison table"
	@echo "  make exp-n200       Sweep n_estimators=200"
	@echo "  make exp-depth15    Sweep max_depth=15"
	@echo ""
	@echo "Testing:"
	@echo "  make test           Run full pytest suite"
	@echo "  make test-cov       Run tests with coverage report"
	@echo ""
	@echo "Benchmarks:"
	@echo "  make benchmark      Compare monolithic vs DVC runtimes"
	@echo "  make test-cache     Validate DVC caching behavior"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   Build Docker image"
	@echo "  make docker-up      Start services with docker-compose"
	@echo "  make docker-test    Run test suite in container"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          Remove generated artifacts"
	@echo "  make clean-all      Full clean including DVC cache"
	@echo ""
