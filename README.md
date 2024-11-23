# graph-gen-annealing

This repository contains code and a Jupyter Notebook for generating graphs and running simulated annealing.

## Files
- `graph_gen_annealing.ipynb`: Main notebook.
- `graph_gen.py`: Python script for graph generation.
- `annealing.py`: Python script for implementing the simulated annealing algorithm.

## Setup and Usage Instructions

### 1. Clone the Repository
First, clone the repository to your local machine:
```bash
git clone <your-github-repo-url>
cd graph-gen-annealing
```

### 2. Install Required Packages
Ensure you have Python 3.x installed. Install the required libraries by running:
```bash
pip install -r requirements.txt
```
> Note: If no `requirements.txt` file exists, manually check the scripts and notebook for necessary libraries and install them using `pip`.

### 3. Understand the Files
- **`graph_gen.py`**: Contains functions for generating graphs. You can use this standalone or as part of the notebook.
- **`annealing.py`**: Contains the simulated annealing algorithm. It is modular and can be imported for standalone use or called from the notebook.
- **`graph_gen_annealing.ipynb`**: Combines the functionalities of `graph_gen.py` and `annealing.py` into an interactive environment for running experiments.

### 4. Open the Jupyter Notebook
Launch the Jupyter Notebook server and open the notebook file:
```bash
jupyter notebook graph_gen_annealing.ipynb
```

### 5. Configure Parameters
Below are the main configurable parameters based on the attached scripts:

#### `graph_gen` Parameters
Below are some of the key parameters found in `graph_gen.py`:
```
num_nodes = 4  # Number of nodes
edge_density = 1  # Edge density
weight_range = (-1, 1)  # Ragne of edge weights
```
Modify these parameters in the corresponding Python files or directly in the Jupyter Notebook to customize the graph generation and annealing process.

#### `annealing.py` Parameters
- **`algorithm`**: Choose the algorithm to use
  - `0`: pSA (Probabilistic Simulated Annealing)
  - `1`: SSA (Simulated Spin Annealing)
- **`gamma, delta`**: Hyperparameters for pSA
  - Example: `gamma = 0.1`, `delta = 10`
- **`rand_type`**: Type of random generation
  - `0`: Normal distribution
  - `1`: Poisson distribution
- **`control`**: Execution mode
  - `0`: Normal mode # I0 chnages from I0_min to I0_max
  - `1`: Static mode # I0 remains as I0_min


### 6. Run the Notebook
Follow these steps:
1. Open the notebook in Jupyter.
2. Execute each cell in sequence by pressing `Shift + Enter`.
3. Observe the results and save outputs as needed.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Contributing
Contributions are welcome! If you have suggestions or improvements, please create an issue or submit a pull request.

