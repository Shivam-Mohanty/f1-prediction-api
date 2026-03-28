VENV = venv
PYTHON = $(VENV)/bin/python3
PIP = $(VENV)/bin/pip

# Default action when you just type 'make'
all: setup

# Create venv and install dependencies
setup: requirements.txt
	test -d $(VENV) || python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt
	touch $(VENV)/bin/activate

# Clean up the environment
clean:
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} +

# Run the API
run: setup
	$(VENV)/bin/uvicorn main:app --reload