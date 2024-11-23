# graph-gen-annealing

This repository contains a Jupyter Notebook for generating graphs and running simulated annealing.

## Files
- `graph_gen_annealing.ipynb`: Main notebook.

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
> Note: If no `requirements.txt` file exists, manually check the notebook for necessary libraries and install them using `pip`.

### 3. Open the Jupyter Notebook
Launch the Jupyter Notebook server and open the notebook file:
```bash
jupyter notebook graph_gen_annealing.ipynb
```

### 4. Configure Parameters
In the notebook, you can adjust the following parameters to customize the graph generation and annealing process:

- **`num_nodes`**: Number of nodes in the graph.
- **`annealing_steps`**: Number of steps for the simulated annealing algorithm.
- **`temperature`**: Initial temperature for annealing.
- **`cooling_rate`**: The rate at which the temperature decreases.

Modify these parameters in the first few cells of the notebook, then execute the notebook to see the results.

### 5. Run the Notebook
Follow these steps:
1. Open the notebook in Jupyter.
2. Execute each cell in sequence by pressing `Shift + Enter`.
3. Observe the results and save outputs as needed.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Contributing
Contributions are welcome! If you have suggestions or improvements, please create an issue or submit a pull request.

