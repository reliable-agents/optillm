.PHONY: style quality

# make sure to test the local checkout in scripts and not the pre-installed one (don't use quotes!)
export PYTHONPATH = optillm

check_dirs := optillm scripts test.py optillm.py

style:
	python -m isort $(check_dirs) setup.py
	ruff format --line-length 119 --target-version py310 $(check_dirs) setup.py

quality:
	python -m isort --check-only $(check_dirs) setup.py
	ruff check --line-length 119 --target-version py310 $(check_dirs)