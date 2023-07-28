python -m venv .env
source .env/bin/activate

pip install --upgrade pip

pip install --upgrade twine
python setup.py sdist
python setup.py bdist_wheel
python -m twine check dist/*

python -m twine upload dist/*
