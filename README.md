# PFL-attack-silmulation

Install the simulation dependencies:
    - dir: (project root) 
    - command: make install

Run the simulation:
    - dir: (project root) 
    - command: make run

Run the hyperparameter search:
    - dir: (project root)
    - DLG only: `python hyperparameter_simulation.py --attack DLG`
    - InvertingGradients only: `python hyperparameter_simulation.py --attack InvertingGradients`
    - both attacks: `python hyperparameter_simulation.py --attack all`
    - override image count: `python hyperparameter_simulation.py --attack DLG --image-count 1`
    - override run count: `python hyperparameter_simulation.py --attack DLG --run-count 1`
    - override both counts: `python hyperparameter_simulation.py --attack DLG --image-count 1 --run-count 1`
    - override results directory: `python hyperparameter_simulation.py --attack DLG --results-dir ./my_results/`
    