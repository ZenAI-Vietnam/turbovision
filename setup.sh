curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --python 3.10 --clear
source .venv/bin/activate
uv pip install pip
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install ultralytics roboflow ipykernel