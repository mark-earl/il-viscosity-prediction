NOTE: Does not work with Python 3.13, stick with Python 3.12
# Setting up virtual env (Windows):
```
mkdir .venv
python -m venv ./.venv
./.venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r ./requirements.txt
<Develop>
deactivate
rm ./.venv -r
```
