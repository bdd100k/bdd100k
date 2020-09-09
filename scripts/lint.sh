python3 -m pylint bdd100k
python3 -m flake8 --docstring-convention google bdd100k
python3 -m mypy --strict bdd100k
python3 -m black --check bdd100k
python3 -m isort -c bdd100k/**/*.py