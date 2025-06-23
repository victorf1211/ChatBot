# ChatBot

## Environment setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Run `make install` to install the required packages, then activate the virtual environment with  
   ` source .venv/bin/activate `
3. Enable pre-commit hooks with `pre-commit install` and run the first full check with `pre-commit run --all-files`
4. Add your environment variables in the .env file.

## Running the project

- `make run_api`: starts the Uvicorn API server.
- `make run_app`: starts the Streamlit web server.
