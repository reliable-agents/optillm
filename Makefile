.PHONY: style quality

# make sure to test the local checkout in scripts and not the pre-installed one (don't use quotes!)
export PYTHONPATH = optillm

check_dirs := optillm scripts setup.py setup.py test.py optillm.py

style:
	ruff check --select I --fix $(check_dirs)
	ruff format $(check_dirs) setup.py 

quality:
	ruff check --select I $(check_dirs) setup.py