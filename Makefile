.PHONY: install poetry-install demo test validate clean

install:
	@echo ">> Installing Cython<3 (eval7 build dep)"
	pip3 install "Cython<3"
	@echo ">> Installing eval7 with --no-build-isolation"
	pip3 install --no-build-isolation eval7==0.1.7
	@echo ">> Installing rest of requirements"
	pip3 install flask numpy scipy treys scikit-learn

poetry-install:
	@echo ">> Creating Poetry venv with local Python"
	poetry env use "$$(pyenv which python)"
	@echo ">> Installing legacy build tools for eval7"
	poetry run python -m pip install setuptools wheel "Cython<3"
	@echo ">> Installing eval7 with --no-build-isolation"
	poetry run python -m pip install --no-build-isolation eval7==0.1.7
	@echo ">> Installing locked dependencies"
	poetry install --no-root

demo:
	python demo.py

test:
	python -m pytest tests/ -q

validate:
	@if [ -z "$(BOT)" ]; then echo "usage: make validate BOT=bots/mybot/bot.py"; exit 1; fi
	python sandbox/validator.py $(BOT)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
