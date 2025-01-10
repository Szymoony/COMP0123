import networkx as nx
import matplotlib.pyplot as plt
import random
import time
from itertools import combinations
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import numpy as np
def save_network_to_csv(G, filename="sbm_network.csv"):
    """
    Saves the network as a CSV file (edge list) for Gephi.
    
    Parameters:
    - G: The networkx graph to save.
    - filename: Name of the output CSV file.
    """
    # Convert the graph to an edge list
    edge_list = list(G.edges())
    
    # Save as a CSV file
    df = pd.DataFrame(edge_list, columns=["Source", "Target"])
    df.to_csv(filename, index=False)
    print(f"Network saved to {filename}")

def generate_sbm_network(G, communities, custom_p_within=None, custom_p_between=None):
    """
    Generates a Stochastic Block Model (SBM) network that matches the properties
    of the input network's community structure, maintaining similar edge count.
    
    Parameters:
    - G: Original network (networkx Graph)
    - communities: List of communities from original network
    - custom_p_within: Optional; custom probability for within-community connections
    - custom_p_between: Optional; custom probability for between-community connections
    
    Returns:
    - G_sbm: A Stochastic Block Model network
    """
    # Calculate community sizes
    sizes = [len(c) for c in communities]
    num_communities = len(communities)
    original_edges = G.number_of_edges()
    
    if custom_p_within is None or custom_p_between is None:
        # Count actual within and between community edges
        within_edges = 0
        between_edges = 0
        within_possible = 0
        between_possible = 0
        
        # Calculate within-community edges and possibilities
        for comm in communities:
            subg = G.subgraph(comm)
            within_edges += subg.number_of_edges()
            n = len(comm)
            within_possible += (n * (n-1)) // 2
        
        # Calculate between-community edges and possibilities
        for i in range(num_communities):
            for j in range(i + 1, num_communities):
                edges_between = sum(1 for u in communities[i] 
                                  for v in communities[j] 
                                  if G.has_edge(u, v))
                between_edges += edges_between
                between_possible += len(communities[i]) * len(communities[j])
        
        # Calculate probabilities to maintain edge proportions
        p_within = within_edges / within_possible if within_possible > 0 else 0
        p_between = between_edges / between_possible if between_possible > 0 else 0
    else:
        p_within = custom_p_within
        p_between = custom_p_between
    
    # Create probability matrix
    block_matrix = [[p_within if i == j else p_between 
                    for j in range(num_communities)] 
                   for i in range(num_communities)]
    
    # Generate the SBM network
    G_sbm = nx.stochastic_block_model(sizes, block_matrix, seed=42)
    
    # Print comparison metrics
    print("\nSBM Network Properties:")
    print(f"Number of communities: {num_communities}")
    print(f"Within-community probability: {p_within:.4f}")
    print(f"Between-community probability: {p_between:.4f}")
    print(f"Original network edges: {original_edges}")
    print(f"SBM network edges: {G_sbm.number_of_edges()}")
    print(f"Original network density: {nx.density(G):.4f}")
    print(f"SBM network density: {nx.density(G_sbm):.4f}")
    
    return G_sbm

# from scipy.stats import percentileofscore

# --- Loading the Dataset ---

def load_graph_data(filepath):
    """Loads graph data from an edgelist file."""
    try:
        G = nx.read_edgelist(filepath, nodetype=int, data=False)
        return G
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None
# def load_graph_data(filepath):
#     """Loads graph data from a CSV file with 'from,to' header."""
#     try:
#         # Read CSV with pandas
#         df = pd.read_csv(filepath)
        
#         # Create networkx graph from edge list
#         # Using df['from'] and df['to'] to get the columns
#         G = nx.from_pandas_edgelist(
#             df,
#             source='source',
#             target='target',
#             create_using=nx.Graph()  # Use Graph() for undirected, DiGraph() for directed
#         )
        
#         return G
#     except FileNotFoundError:
#         print(f"Error: File not found at {filepath}")
#         return None
#     except Exception as e:
#         print(f"An error occurred while loading the data: {e}")
#         print("Detailed error:", str(e))  # More detailed error message
#         return None
# --- Basic Network Analysis ---

def analyze_network(G):
    """Analyzes basic properties of the network and visualizes degree distribution."""
    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())
    avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
    print("Average degree:", avg_degree)
    print("Density:", nx.density(G))
    print("Is connected:", nx.is_connected(G))

    if nx.is_connected(G):
        print("Diameter:", nx.diameter(G))
        print("Average shortest path length:", nx.average_shortest_path_length(G))
    else:
        print("Diameter: Network is not connected, calculating diameter of the largest component")
        largest_cc = max(nx.connected_components(G), key=len)
        G_largest = G.subgraph(largest_cc)
        print("Diameter of largest connected component:", nx.diameter(G_largest))

    print("Average clustering coefficient:", nx.average_clustering(G))

    # Degree Distribution Analysis and Visualization
    degrees = [G.degree(n) for n in G.nodes()]
    degree_df = pd.DataFrame(degrees, columns=['Degree'])

    # Summary statistics
    print("\nDegree Distribution Summary Statistics:")
    print(degree_df.describe())

    # Visualization
    plt.figure(figsize=(10, 5))
    sns.histplot(degree_df['Degree'], bins=50, kde=True)
    plt.title("Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.savefig("degree_distribution.png")
    plt.show()

    return degree_df

# --- Community Detection (Louvain) ---

def detect_communities(G):
    """
    Detects communities using the Louvain algorithm and provides detailed analysis.
    """
    try:
        communities = nx.community.louvain_communities(G)
        print("\n--- Community Detection Results ---")
        print(f"Number of communities detected: {len(communities)}")

        # Community Size Analysis
        community_sizes = [len(c) for c in communities]
        print("\nCommunity Size Distribution:")
        print(f"  Min: {min(community_sizes)}")
        print(f"  Max: {max(community_sizes)}")
        print(f"  Mean: {sum(community_sizes) / len(community_sizes):.2f}")
        print(f"  Median: {sorted(community_sizes)[len(community_sizes) // 2]}")

        # Visualize community size distribution
        plt.figure(figsize=(8, 4))
        sns.histplot(community_sizes, bins=50, kde=False)
        plt.title("Community Size Distribution")
        plt.xlabel("Community Size")
        plt.ylabel("Frequency")
        plt.savefig("community_size_distribution.png")
        plt.show()

        # Largest Communities Analysis
        largest_communities = sorted(communities, key=len, reverse=True)[:5]
        print("\nLargest 5 Communities:")
        for i, community in enumerate(largest_communities):
            subgraph = G.subgraph(community)
            density = nx.density(subgraph)
            print(f"  Community {i+1}:")
            print(f"    Size: {len(community)}")
            print(f"    Density: {density:.4f}")
            if nx.is_connected(subgraph):
                print(f"    Diameter: {nx.diameter(subgraph)}")
                print(f"    Average Shortest Path Length: {nx.average_shortest_path_length(subgraph):.2f}")
            else:
                print(f"    Diameter: Not connected")
                print(f"    Average Shortest Path Length: Not connected")
            print(f"    Average Clustering Coefficient: {nx.average_clustering(subgraph):.4f}")

        return communities
    except Exception as e:
        print(f"An error occurred during community detection: {e}")
        return None

# --- Path Length Calculation ---

def calculate_path_length_between_communities(G, communities, num_pairs=10, sample_size=20):
    """Calculates average path length between random community pairs."""
    community_pairs = []
    if len(communities) >= 2:
        for i in range(num_pairs):
            pair = random.sample(range(len(communities)), 2)
            community_pairs.append((pair[0], pair[1]))

    path_lengths = []
    for pair in community_pairs:
        community1 = list(communities[pair[0]])
        community2 = list(communities[pair[1]])

        total_path_length = 0
        num_paths = 0

        # Sample from the original graph G
        sample_community1 = random.sample(community1, min(sample_size, len(community1)))
        sample_community2 = random.sample(community2, min(sample_size, len(community2)))

        for u in sample_community1:
            for v in sample_community2:
                if u != v and nx.has_path(G, u, v):
                    total_path_length += nx.shortest_path_length(G, u, v)
                    num_paths += 1

        if num_paths > 0:
            avg_path_length = total_path_length / num_paths
            path_lengths.append((pair, avg_path_length))
        else:
            path_lengths.append((pair, float('inf')))

    return path_lengths

# --- Betweenness Centrality Calculation ---

def calculate_betweenness_centrality(G, visualize=True):
    """
    Calculates betweenness centrality for all nodes and optionally visualizes the distribution.
    """
    print("Calculating betweenness centrality...")
    betweenness = nx.betweenness_centrality(G)

    if visualize:
        # Convert to DataFrame for easier handling
        betweenness_df = pd.DataFrame(list(betweenness.items()), columns=['Node', 'Betweenness Centrality'])

        # Summary statistics
        print("\nBetweenness Centrality Summary Statistics:")
        print(betweenness_df.describe())

        # Visualization
        plt.figure(figsize=(10, 5))
        sns.histplot(betweenness_df['Betweenness Centrality'], bins=50, kde=True)
        plt.title('Betweenness Centrality Distribution')
        plt.xlabel('Betweenness Centrality')
        plt.ylabel('Frequency')
        plt.savefig('betweenness_centrality_distribution.png')
        plt.show()

    return betweenness

# --- Node Removal ---

def remove_top_betweenness_nodes(G, percentage):
    """Removes a specified percentage of nodes with the highest betweenness centrality."""
    betweenness = calculate_betweenness_centrality(G)
    num_nodes_to_remove = int(len(G.nodes()) * percentage)
    sorted_nodes = sorted(betweenness.items(), key=lambda item: item[1], reverse=True)
    nodes_to_remove = [node for node, centrality in sorted_nodes[:num_nodes_to_remove]]

    G_copy = G.copy()
    G_copy.remove_nodes_from(nodes_to_remove)
    return G_copy

# --- Enhanced Hidden Bridge Identification ---

def identify_hidden_bridges(G, communities, num_pairs=80, sample_size=100, impact_threshold=0.001, betweenness_percentile_threshold=98):
    """
    Modified to only check nodes that lie on shortest paths between sampled community pairs
    """
    print("\n--- Identifying Hidden Bridges ---")
    start_time = time.time()

    betweenness_centrality = calculate_betweenness_centrality(G, visualize=False)
    betweenness_threshold = pd.Series(betweenness_centrality).quantile(betweenness_percentile_threshold / 100)

    community_pairs = list(combinations(range(len(communities)), 2))
    # if num_pairs < len(community_pairs):
    #     community_pairs = random.sample(community_pairs, num_pairs)

    hidden_bridges_df = pd.DataFrame(columns=['Node', 'Betweenness Centrality', 'Impact on Path Length', 'Community Pair', 'Degree', 'Clustering Coefficient'])

    for pair in tqdm(community_pairs, desc="Analyzing community pairs"):
        # Sample nodes from each community
        community1 = list(communities[pair[0]])
        community2 = list(communities[pair[1]])
        # sample1 = random.sample(community1, min(sample_size, len(community1)))
        # sample2 = random.sample(community2, min(sample_size, len(community2)))
        sample1 = community1
        sample2 = community2

        # Find all nodes that lie on shortest paths between sampled nodes
        nodes_to_consider = set()
        initial_path_length = 0
        num_paths = 0
        
        for u in sample1:
            for v in sample2:
                if u != v and nx.has_path(G, u, v):
                    path = nx.shortest_path(G, u, v)
                    # print(path)
                    nodes_to_consider.update(path[1:-1])  # Exclude endpoints
                    initial_path_length += len(path) - 1
                    num_paths += 1

        initial_avg_path_length = initial_path_length / num_paths if num_paths > 0 else float('inf')
        # print(f"nodes to consider: {nodes_to_consider}")
        # Only check nodes that are on shortest paths and below betweenness threshold
        G_copy = G.copy()
        for node in nodes_to_consider:
            # if betweenness_centrality[node] <= betweenness_threshold:
                if node in G_copy: 
                    G_copy.remove_node(node)
                    total_length = 0
                    valid_paths = 0
                    
                    for u in sample1:
                        for v in sample2:
                            if u != node and v != node and nx.has_path(G_copy, u, v) and u != v:
                                total_length += nx.shortest_path_length(G_copy, u, v)
                                valid_paths += 1
                            # if u != v and nx.has_path(G_copy, u, v):
                            #     total_length += nx.shortest_path_length(G_copy, u, v)
                            #     valid_paths += 1
                    
                    avg_path_length = total_length / valid_paths if valid_paths > 0 else float('inf')
                    
                    if initial_avg_path_length != float('inf'):
                        impact = (avg_path_length - initial_avg_path_length) / initial_avg_path_length
                        
                        if impact > impact_threshold:
                            new_row = pd.DataFrame({
                                'Node': [node],
                                'Betweenness Centrality': [betweenness_centrality[node]],
                                'Impact on Path Length': [impact],
                                'Community Pair': [f"{pair[0]}-{pair[1]}"],
                                'Degree': [G.degree(node)],
                                'Clustering Coefficient': [nx.clustering(G, node)]
                            })
                            hidden_bridges_df = pd.concat([hidden_bridges_df, new_row], ignore_index=True)
                    
                    G_copy = G.copy()  # Reset for next node

    end_time = time.time()
    print(f"Time taken for hidden bridge identification: {end_time - start_time:.2f} seconds")

    if not hidden_bridges_df.empty:
        hidden_bridges_df.sort_values(by='Impact on Path Length', ascending=False, inplace=True)
        print("\nTop Hidden Bridges Identified:")
        print(hidden_bridges_df)
        
        # hidden_bridges_df['Betweenness Percentile'] = hidden_bridges_df['Betweenness Centrality'].apply(
        #     lambda x: percentileofscore(list(betweenness_centrality.values()), x))
        hidden_bridges_df.to_csv('hidden_bridges_facebook_random.csv', index=False)
    else:
        print("No hidden bridges identified with the given criteria.")

    return hidden_bridges_df

def calculate_avg_path_length_for_pair(G, communities, pair, sample_size):
    """
    Calculates the average path length for a community pair.
    """
    community1 = list(communities[pair[0]])
    community2 = list(communities[pair[1]])

    sample_community1 = random.sample(community1, min(sample_size, len(community1)))
    sample_community2 = random.sample(community2, min(sample_size, len(community2)))

    total_path_length = 0
    num_paths = 0

    for u in sample_community1:
        for v in sample_community2:
            if u != v and nx.has_path(G, u, v):
                total_path_length += nx.shortest_path_length(G, u, v)
                num_paths += 1

    return total_path_length / num_paths if num_paths > 0 else float('inf')

# --- Visualization of Hidden Bridges ---

# def visualize_hidden_bridges(G, hidden_bridges_df, num_bridges_to_visualize=5):
#     """
#     Visualizes the top hidden bridge nodes and their connections within the network.
#     """
#     if hidden_bridges_df.empty:
#         print("No hidden bridges to visualize.")
#         return

#     top_bridges = hidden_bridges_df.head(num_bridges_to_visualize)

#     for index, row in top_bridges.iterrows():
#         node = row['Node']
#         community_pair = row['Community Pair']
#         impact = row['Impact on Path Length']

#         # Create a subgraph for visualization
#         subgraph_nodes = set()
#         for community in map(int, community_pair.split('-')):
#             subgraph_nodes.update(communities[community])

#         # Include neighbors of the hidden bridge node
#         neighbors = set(G.neighbors(node))
#         subgraph_nodes.update(neighbors)
#         subgraph_nodes.add(node)  # Ensure the hidden bridge node is included

#         subgraph = G.subgraph(subgraph_nodes)

#         # Draw the subgraph
#         plt.figure(figsize=(10, 8))
#         pos = nx.spring_layout(subgraph, seed=42)  # Seed for reproducibility
#         nx.draw(subgraph, pos, with_labels=True, node_color='skyblue', node_size=800, edge_color='gray')
#         nx.draw_networkx_nodes(subgraph, pos, nodelist=[node], node_color='red', node_size=800)  # Highlight the bridge node

#         plt.title(f"Hidden Bridge Node: {node}\nCommunity Pair: {community_pair}\nImpact: {impact:.4f}")
#         plt.savefig(f"hidden_bridge_node_{node}.png")
#         plt.show()

# --- Main Execution ---

if __name__ == "__main__":
    filepath = "/Users/szymongoralczuk/Downloads/facebook_combined.txt"  # Replace with your file path
    # filepath = "/Users/szymongoralczuk/Documents/Complex Network/musae_RU_edges.csv"
    G = load_graph_data(filepath)

    if G:
        degree_df = analyze_network(G)
        communities = detect_communities(G)

        if communities:
            print("\n--- Preliminary 'Hidden Bridge' Analysis ---")
            path_lengths = calculate_path_length_between_communities(G, communities, num_pairs=5)
            for pair, avg_path_length in path_lengths:
                print(f"Average path length between communities {pair[0]} and {pair[1]}: {avg_path_length:.2f}")

            # --- Betweenness Centrality and Node Removal ---
            # print("\n--- Removing Top Betweenness Nodes ---")
            # start_time = time.time()
            # G_reduced = remove_top_betweenness_nodes(G, 0.01)  # Remove top 1%
            # end_time = time.time()
            # print(f"Time taken for removing top betweenness nodes: {end_time - start_time:.2f} seconds")

            # print("\n--- Analyzing Network After Node Removal ---")

            # # Update communities to only include nodes present in G_reduced
            # communities_reduced = [[node for node in community if node in G_reduced] for community in communities]

            # path_lengths_after = calculate_path_length_between_communities(G_reduced, communities_reduced, num_pairs=5)
            # for pair, avg_path_length in path_lengths_after:
            #     print(f"Average path length between communities {pair[0]} and {pair[1]}: {avg_path_length:.2f}")
            
            # Identify and Analyze Hidden Bridges
            # hidden_bridges_df = identify_hidden_bridges(G, communities)
                        # Generate an SBM network based on the Facebook network's community structure
            # print("\n--- Generating SBM Network ---")
            G_sbm = generate_sbm_network(G, communities)
            
            # # Analyze the SBM network
            # print("\n--- Analyzing SBM Network ---")
            degree_df_sbm = analyze_network(G_sbm)
            communities_sbm = detect_communities(G_sbm)
            save_network_to_csv(G_sbm, filename="sbm_network.csv")

            # if communities_sbm:
            #     print("\n--- 'Hidden Bridge' Analysis in SBM Network ---")
            #     path_lengths_sbm = calculate_path_length_between_communities(G_sbm, communities_sbm, num_pairs=5)
            #     for pair, avg_path_length in path_lengths_sbm:
            #         print(f"Average path length between communities {pair[0]} and {pair[1]}: {avg_path_length:.2f}")

            #     # Identify and Analyze Hidden Bridges in the SBM network
            #     hidden_bridges_df_sbm = identify_hidden_bridges(G_sbm, communities_sbm)
                
            #     # Compare the results
            #     # print("\n--- Comparison of Hidden Bridges ---")
            #     # print("Facebook Network:")
            #     # print(hidden_bridges_df.describe())
            #     print("\nSBM Network:")
            #     print(hidden_bridges_df_sbm.describe())


            
            # Visualize Hidden Bridges
            # if not hidden_bridges_df.empty:
            #     visualize_hidden_bridges(G, hidden_bridges_df)

    print("Done.")