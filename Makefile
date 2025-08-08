# Quantum GPT Makefile

.PHONY: install test train clean example help evaluate report

help:
	@echo "Quantum-Enhanced GPT Commands:"
	@echo "  make install    - Install all dependencies"
	@echo "  make data       - Download Shakespeare dataset"
	@echo "  make test       - Run comprehensive tests"
	@echo "  make train      - Train quantum GPT model"
	@echo "  make generate   - Generate text from trained model"
	@echo "  make evaluate   - Evaluate components of the model"
	@echo "  make report     - Train classical & quantum and write models/results.md"
	@echo "  make clean      - Clean up generated files"

install:
	pip install -r requirements.txt

data:
	curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o data/input.txt

test:
	python src/test_quantum_gpt.py

train:
	python src/train_quantum_gpt.py

generate:
	python src/train_quantum_gpt.py generate "First Citizen:"

clean:
	rm -f quantum_gpt_model.pth
	rm -f *.pyc
	rm -rf __pycache__
	rm -rf .pytest_cache

evaluate:
	python src/evaluavate_components.py

report:
	python src/train_both_and_report.py