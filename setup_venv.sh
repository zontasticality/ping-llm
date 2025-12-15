# 2. Create virtual environment
uv venv --python 3.12 --seed .venv
source .venv/bin/activate

# 3. Install dependencies in editable mode
# install the tpu package
# uv pip install -e .[tpu] --resolution=lowest
# or install the gpu package by running the following line
uv pip install -e .[cuda12] --resolution=lowest
install_maxtext_github_deps