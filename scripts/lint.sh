python3 -m black bdd100k
python3 -m isort bdd100k
python3 -m pylint bdd100k
python3 -m flake8 --docstring-convention google bdd100k
python3 -m mypy --install-types --strict --allow-untyped-calls --show-error-codes bdd100k
