import os

# File path
file_path = "/mnt/data/graph_gen_annealing.ipynb"
repo_name = "graph-gen-annealing"

# Creating a README.md content
readme_content = f"""# {repo_name}

This repository contains a Jupyter Notebook for generating graphs and running simulated annealing.

## Files
- `graph_gen_annealing.ipynb`: Main notebook.

## Usage
Open the notebook using Jupyter to explore graph generation and annealing algorithms.

## Requirements
Ensure you have the following installed:
- Python 3.x
- Jupyter Notebook
- Required Python libraries (as listed in the notebook)

## License
Feel free to customize the license type.
"""

# Writing the README.md
readme_path = os.path.join("/mnt/data", "README.md")
with open(readme_path, "w") as f:
    f.write(readme_content)

# Git instructions for the user
instructions = """
# GitHub Setup Instructions

1. Initialize a new repository on GitHub (name: graph-gen-annealing).
2. Clone the repository to your local machine.
3. Add the files to the repository and push them:

```bash
git init
git add graph_gen_annealing.ipynb README.md
git commit -m "Initial commit"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
