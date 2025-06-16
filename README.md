# DevBot Backend Rewrite

We figured it would be much easier using Python for AI/ML integrations. Hence we are rewriting the backend and flows of the Devbot project in Python using the Robyn Web Framework.

## Setup

Ensure that you have UV installed, you can learn to do so [here](https://docs.astral.sh/uv/getting-started/installation/).
1. Run `uv sync`. Should download the correct python version, add required dependencies. You can then use `uv add <PACKAGE_NAME>` to add a package.
2. Run `uv run src/app.py` to run the Robyn app.
