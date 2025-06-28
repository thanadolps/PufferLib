```
uv venv --python 3.8
source .venv/bin/activate
python --version
```

smb want 3.8 (https://github.com/Kautenja/gym-super-mario-bros/issues/135), but pufferlib want >=3.9...

```
uv pip install setuptools wheel Cython numpy torch click requests pyro-ppl opencv-python
```

```
uv pip install '.[train,smb]' --no-build-isolation
```
