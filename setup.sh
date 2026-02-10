# curl -LsSf https://astral.sh/uv/install.sh | sh
# source $HOME/.local/bin/env
# uv venv --python 3.10 --clear
# require cude 13.0
source .venv/bin/activate
uv pip install pip
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install "cuml-cu13==26.2.*"
uv pip install -r requirements.txt