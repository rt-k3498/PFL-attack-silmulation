.PHONY: run run-hyperparameter-simulation install

run: 
	python simulation.py

run-hyperparameter-simulation:
	python hyperparameter_simulation.py

install:
	pip install -r requirements.txt