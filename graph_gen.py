import networkx as nx
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import csv
import os

def generate_maxcut_problem(num_nodes, edge_density=0.5, weight_range=(-10, 10)):
    """
    Function to generate a MAX-CUT problem of arbitrary size (excluding weight 0)

    Args:
        num_nodes (int): Number of nodes
        edge_density (float): Density of edges (0 to 1)
        weight_range (tuple): Range of edge weights (min, max)

    Returns:
        G (networkx.Graph): Graph representing the MAX-CUT problem
    """
    # Generate a random graph based on edge density
    G = nx.Graph()
    nodes = list(range(num_nodes))
    G.add_nodes_from(nodes)
    
    # Randomly generate edges and assign weights
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.rand() < edge_density:
                # Generate weights excluding 0
                weight = np.random.choice(
                    [x for x in range(weight_range[0], weight_range[1]+1) if x != 0]
                )
                G.add_edge(i, j, weight=weight)
    
    return G

def maxcut_optimal_value(G):
    """
    Function to compute the optimal value for a MAX-CUT problem (brute force)

    Args:
        G (networkx.Graph): Graph representing the MAX-CUT problem

    Returns:
        optimal_value (int): Optimal cut value
        optimal_cut (tuple): Optimal cut (pair of node sets)
    """
    nodes = list(G.nodes)
    num_nodes = len(nodes)
    max_value = float('-inf')
    best_cut = None

    # Try all possible partitions (brute force)
    for i in range(1, num_nodes // 2 + 1):
        for subset in combinations(nodes, i):
            set1 = set(subset)
            set2 = set(nodes) - set1
            cut_value = sum(G[u][v]['weight'] for u in set1 for v in set2 if G.has_edge(u, v))
            if cut_value > max_value:
                max_value = cut_value
                best_cut = (set1, set2)

    return max_value, best_cut

def save_graph_as_csv(G, filename):
    """
    Function to save the graph in adjacency matrix format as a CSV

    Args:
        G (networkx.Graph): Graph
        filename (str): Name of the CSV file to save
    """
    # Get adjacency matrix as a NumPy array
    adjacency_matrix = nx.to_numpy_array(G, weight='weight', dtype=int)

    # Write to a CSV file
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write data
        writer.writerows(adjacency_matrix)

def save_negated_graph_as_csv(G, filename):
    """
    Function to save a graph with negated edge weights as a CSV

    Args:
        G (networkx.Graph): Graph
        filename (str): Name of the CSV file to save
    """
    # Create a copy of the graph and negate the edge weights
    negated_G = G.copy()
    for u, v, data in negated_G.edges(data=True):
        data['weight'] = -data['weight']

    # Get adjacency matrix as a NumPy array
    adjacency_matrix = nx.to_numpy_array(negated_G, weight='weight', dtype=int)

    # Write to a CSV file
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(adjacency_matrix)

def save_nonzero_elements_with_metadata_as_txt(G, num_nodes, optimal_value, filename):
    """
    Function to save edge information of the graph in "row column value" format to a TXT,
    adding num_nodes and optimal_value at the beginning

    Args:
        G (networkx.Graph): Graph
        num_nodes (int): Number of nodes
        optimal_value (int): Optimal cut value
        filename (str): Name of the TXT file to save
    """
    with open(filename, mode='w') as file:
        # Write num_nodes and optimal_value first
        file.write(f"{num_nodes}\n")
        file.write(f"{optimal_value}\n")
        
        # Write non-zero edge information in "row column value" format
        for u, v, data in G.edges(data=True):
            weight = data['weight']
            if weight != 0:
                file.write(f"{u} {v} {weight}\n")


# Create the directory if it does not exist
os.makedirs('./matrix', exist_ok=True)

# Generate the problem
num_nodes = 4  # Number of nodes
edge_density = 1  # Edge density
weight_range = (-1, 1)  # Range of edge weights (integers)
G = generate_maxcut_problem(num_nodes, edge_density, weight_range)

# Compute the optimal value
optimal_value, optimal_cut = maxcut_optimal_value(G)

# Display results
print("Generated Graph:")
for u, v, data in G.edges(data=True):
    print(f"  {u}-{v}: weight = {data['weight']}")

print("\nOptimal Cut Value:", optimal_value)
print("Optimal Cut:", optimal_cut)

# Draw the graph using Circular Layout
pos = nx.circular_layout(G)  # Arrange nodes in a circular layout
nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=500)
nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'))
plt.show()

# Save the graph as a CSV
csv_filename = f'./matrix/graph_N{num_nodes}_WL{weight_range[0]}_WH{weight_range[1]}_D{edge_density}.csv'
save_graph_as_csv(G, csv_filename)

print(f"Graph saved as adjacency matrix in CSV file '{csv_filename}'.")

# Example usage
txt_filename_with_metadata = f'./matrix/sparse_N{num_nodes}_WL{weight_range[0]}_WH{weight_range[1]}_D{edge_density}.txt'
save_nonzero_elements_with_metadata_as_txt(G, num_nodes, optimal_value, txt_filename_with_metadata)

print(f"Graph saved with metadata and non-zero edge information in TXT file '{txt_filename_with_metadata}'.")

# Save negated weights as a CSV
negated_csv_filename = f'./matrix/J_N{num_nodes}_WL{weight_range[0]}_WH{weight_range[1]}_D{edge_density}.csv'
save_negated_graph_as_csv(G, negated_csv_filename)

print(f"Negated weights saved in CSV file '{negated_csv_filename}'.")
