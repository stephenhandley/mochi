# Running on MacOS

## Install / Setup
First had to comment PyTorch deps in [pyproject.toml](./pyproject.toml) and [requirements.txt](./requirements.txt) to get PyTorch nightly installed for MPS support per [Apple's MPS docs](https://developer.apple.com/metal/pytorch/).

```
uv venv
source .venv/bin/activate
uv pip install setuptools
uv pip install -e . --no-build-isolation # Looks like flash attention support is CUDA only 
uv pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
uv run python ./scripts/download_weights.py weights/
uv run python ./demos/gradio_ui.py --model_dir weights/
```

## Device contraints

Had to reduce the frame size and count due to resource constraints on my machine

Here's the changes to the defaults that worked for me in Gradio.
```
width = 640   # Requires: width % 16 == 0
height = 384  # Requires: height % 16 == 0
frames = 49   # Requires: frames = (n * 6) + 1 where n is some integer
