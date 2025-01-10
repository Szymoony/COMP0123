import networkx as nx
import matplotlib.pyplot as plt
import random
import time
from itertools import combinations

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

# --- Basic Network Analysis ---

def analyze_network(G):
    """Analyzes basic properties of the network."""
    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())
    print("Average degree:", sum(dict(G.degree()).values()) / G.number_of_nodes())
    print("Density:", nx.density(G))
    print("Is connected:", nx.is_connected(G))

    if nx.is_connected(G):
        print("Diameter:", nx.diameter(G))
    else:
        print("Diameter: Network is not connected, calculating diameter of the largest component")
        largest_cc = max(nx.connected_components(G), key=len)
        G_largest = G.subgraph(largest_cc)
        print("Diameter of largest connected component:", nx.diameter(G_largest))

    print("Average clustering coefficient:", nx.average_clustering(G))

    # Degree Distribution
    degrees = [G.degree(n) for n in G.nodes()]
    plt.hist(degrees, bins=50)
    plt.title("Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.show()

# --- Community Detection (Louvain) ---

def detect_communities(G):
    """Detects communities using the Louvain algorithm."""
    try:
        communities = nx.community.louvain_communities(G)
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

def calculate_betweenness_centrality(G):
    """Calculates betweenness centrality for all nodes."""
    betweenness = nx.betweenness_centrality(G)
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

# --- Hidden Bridge Identification ---

def identify_hidden_bridges(G, communities, num_pairs=5, sample_size=10):
    """Identifies potential hidden bridge nodes."""
    community_pairs = list(combinations(range(len(communities)), 2))
    if num_pairs < len(community_pairs):
        community_pairs = random.sample(community_pairs, num_pairs)

    hidden_bridges = {pair: [] for pair in community_pairs}
    initial_path_lengths = {}

    # Measure initial path lengths
    for pair in community_pairs:
        community1 = list(communities[pair[0]])
        community2 = list(communities[pair[1]])

        # Sample nodes from the original graph
        sample_community1 = random.sample(community1, min(sample_size, len(community1)))
        sample_community2 = random.sample(community2, min(sample_size, len(community2)))

        total_path_length = 0
        num_paths = 0

        for u in sample_community1:
            for v in sample_community2:
                if u != v and nx.has_path(G, u, v):
                    total_path_length += nx.shortest_path_length(G, u, v)
                    num_paths += 1

        initial_path_lengths[pair] = total_path_length / num_paths if num_paths > 0 else float('inf')

    # Analyze impact of node removal
    for pair in community_pairs:
        community1 = set(communities[pair[0]])
        community2 = set(communities[pair[1]])
        nodes_to_consider = community1.union(community2)

        for node in nodes_to_consider:
            G_copy = G.copy()
            if node in G_copy:  # Check if node exists before removal
                G_copy.remove_node(node)

                # Sample nodes from the original graph
                sample_community1 = random.sample(list(community1), min(sample_size, len(community1)))
                sample_community2 = random.sample(list(community2), min(sample_size, len(community2)))

                total_path_length = 0
                num_paths = 0

                for u in sample_community1:
                    for v in sample_community2:
                        if u != v and u in G_copy and v in G_copy and nx.has_path(G_copy, u, v):
                            total_path_length += nx.shortest_path_length(G_copy, u, v)
                            num_paths += 1

                if num_paths > 0:
                    avg_path_length = total_path_length / num_paths
                    if initial_path_lengths[pair] != float('inf') and avg_path_length > initial_path_lengths[pair]:
                        hidden_bridges[pair].append((node, avg_path_length - initial_path_lengths[pair]))
                elif initial_path_lengths[pair] != float('inf'):
                    hidden_bridges[pair].append((node, float('inf')))

    return hidden_bridges

# --- Main Execution ---

if __name__ == "__main__":
    filepath = "/Users/szymongoralczuk/Downloads/facebook_combined.txt"  # Replace with your file path
    G = load_graph_data(filepath)

    if G:
        analyze_network(G)
        communities = detect_communities(G)

        if communities:
            print("\n--- Community Detection Results ---")
            print(f"Number of communities detected: {len(communities)}")

            print("\n--- Preliminary 'Hidden Bridge' Analysis ---")
            path_lengths = calculate_path_length_between_communities(G, communities, num_pairs=5)
            for pair, avg_path_length in path_lengths:
                print(f"Average path length between communities {pair[0]} and {pair[1]}: {avg_path_length:.2f}")

            # --- Betweenness Centrality and Node Removal ---
            print("\n--- Removing Top Betweenness Nodes ---")
            start_time = time.time()
            G_reduced = remove_top_betweenness_nodes(G, 0.01)  # Remove top 5%
            end_time = time.time()
            print(f"Time taken for removing top betweenness nodes: {end_time - start_time:.2f} seconds")

            print("\n--- Analyzing Network After Node Removal ---")

            # Update communities to only include nodes present in G_reduced
            communities_reduced = [[node for node in community if node in G_reduced] for community in communities]

            path_lengths_after = calculate_path_length_between_communities(G_reduced, communities_reduced, num_pairs=5)
            for pair, avg_path_length in path_lengths_after:
                print(f"Average path length between communities {pair[0]} and {pair[1]}: {avg_path_length:.2f}")

            # --- Hidden Bridge Identification ---
            print("\n--- Identifying Hidden Bridges ---")
            start_time = time.time()
            hidden_bridge_nodes = identify_hidden_bridges(G, communities, num_pairs=5)
            end_time = time.time()
            print(f"Time taken for hidden bridge identification: {end_time - start_time:.2f} seconds")

            for pair, bridges in hidden_bridge_nodes.items():
                if bridges:
                    print(f"\nPotential hidden bridges between communities {pair[0]} and {pair[1]}:")
                    sorted_bridges = sorted(bridges, key=lambda x: x[1], reverse=True)
                    for node, impact in sorted_bridges[:10]:
                        # Check if the node exists in the original graph before calculating betweenness centrality
                        if node in G:
                            betweenness = nx.betweenness_centrality(G)[node]
                            print(f"  Node: {node}, Betweenness Centrality: {betweenness:.4f}, Impact on Path Length: {impact}")
                        else:
                            print(f"  Node: {node} (no longer in graph), Impact on Path Length: {impact}")

    print("Done.")