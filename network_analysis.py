#!/usr/bin/env python3
"""
Network Analysis Module

This module provides functionality for loading, analyzing, and visualizing complex networks.
It includes basic network metrics calculation, community detection, and visualization tools.
"""

import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NetworkAnalyzer:
    """Class for analyzing complex networks."""
    
    def __init__(self, graph: nx.Graph):
        """
        Initialize NetworkAnalyzer with a networkx graph.
        
        Args:
            graph: NetworkX graph object to analyze
        """
        self.G = graph
        self.communities = None
        
    @classmethod
    def from_file(cls, filepath: str) -> 'NetworkAnalyzer':
        """
        Create NetworkAnalyzer instance from an edge list file.
        
        Args:
            filepath: Path to edge list file
            
        Returns:
            NetworkAnalyzer instance
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If file format is invalid
        """
        try:
            G = nx.read_edgelist(filepath, nodetype=int)
            return cls(G)
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading network: {str(e)}")
            raise ValueError(f"Invalid file format: {str(e)}")

    def basic_analysis(self) -> Dict:
        """
        Perform basic network analysis.
        
        Returns:
            Dictionary containing basic network metrics
        """
        metrics = {
            'nodes': self.G.number_of_nodes(),
            'edges': self.G.number_of_edges(),
            'density': nx.density(self.G),
            'avg_clustering': nx.average_clustering(self.G),
            'is_connected': nx.is_connected(self.G)
        }
        
        if metrics['is_connected']:
            metrics['diameter'] = nx.diameter(self.G)
            metrics['avg_path_length'] = nx.average_shortest_path_length(self.G)
        else:
            largest_cc = max(nx.connected_components(self.G), key=len)
            G_largest = self.G.subgraph(largest_cc)
            metrics['diameter_largest_cc'] = nx.diameter(G_largest)
            metrics['avg_path_length_largest_cc'] = nx.average_shortest_path_length(G_largest)
            
        return metrics

    def analyze_degree_distribution(self) -> pd.DataFrame:
        """
        Analyze degree distribution of the network.
        
        Returns:
            DataFrame containing degree distribution data
        """
        degrees = [self.G.degree(n) for n in self.G.nodes()]
        return pd.DataFrame(degrees, columns=['Degree'])

    def detect_communities(self) -> List[set]:
        """
        Detect communities using Louvain algorithm.
        
        Returns:
            List of sets, where each set contains nodes in a community
        """
        try:
            self.communities = list(nx.community.louvain_communities(self.G))
            return self.communities
        except Exception as e:
            logger.error(f"Error in community detection: {str(e)}")
            raise

    def visualize_network(self, output_path: Optional[str] = None):
        """
        Visualize the network with community structure if available.
        
        Args:
            output_path: Optional path to save the visualization
        """
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.G, k=1/np.sqrt(self.G.number_of_nodes()))
        
        if self.communities:
            # Color nodes by community
            colors = [f'C{i}' for i in range(len(self.communities))]
            for idx, community in enumerate(self.communities):
                nx.draw_networkx_nodes(self.G, pos, 
                                     nodelist=list(community),
                                     node_color=colors[idx],
                                     node_size=100,
                                     alpha=0.6)
        else:
            nx.draw_networkx_nodes(self.G, pos, 
                                 node_color='lightblue',
                                 node_size=100,
                                 alpha=0.6)
            
        nx.draw_networkx_edges(self.G, pos, alpha=0.2)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

def analyze_high_betweenness_removal(self, percentage: float = 0.01) -> Dict:
        """
        Analyze the impact of removing top betweenness nodes and compare with random removal.
        
        Args:
            percentage: Percentage of nodes to remove (default: 0.01 for 1%)
            
        Returns:
            Dictionary containing comparison metrics
        """
        # Store original metrics
        original_metrics = self.basic_analysis()
        num_nodes_to_remove = int(len(self.G.nodes()) * percentage)
        
        # Calculate betweenness centrality
        betweenness = nx.betweenness_centrality(self.G)
        
        # Remove top betweenness nodes
        top_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:num_nodes_to_remove]
        G_high_removed = self.G.copy()
        G_high_removed.remove_nodes_from([node for node, _ in top_nodes])
        
        # Calculate metrics after high betweenness removal
        high_removed_metrics = {
            'avg_path_length': nx.average_shortest_path_length(G_high_removed) if nx.is_connected(G_high_removed) 
                              else float('inf'),
            'num_components': nx.number_connected_components(G_high_removed),
            'largest_component_size': len(max(nx.connected_components(G_high_removed), key=len)),
            'clustering_coefficient': nx.average_clustering(G_high_removed)
        }
        
        # Random node removal for comparison
        random_nodes = random.sample(list(self.G.nodes()), num_nodes_to_remove)
        G_random_removed = self.G.copy()
        G_random_removed.remove_nodes_from(random_nodes)
        
        # Calculate metrics after random removal
        random_removed_metrics = {
            'avg_path_length': nx.average_shortest_path_length(G_random_removed) if nx.is_connected(G_random_removed)
                              else float('inf'),
            'num_components': nx.number_connected_components(G_random_removed),
            'largest_component_size': len(max(nx.connected_components(G_random_removed), key=len)),
            'clustering_coefficient': nx.average_clustering(G_random_removed)
        }
        
        # Calculate changes
        results = {
            'high_betweenness_removal': {
                'path_length_change': ((high_removed_metrics['avg_path_length'] - original_metrics['avg_path_length']) 
                                     / original_metrics['avg_path_length'] * 100),
                'num_components': high_removed_metrics['num_components'],
                'largest_component_change': ((high_removed_metrics['largest_component_size'] - self.G.number_of_nodes())
                                           / self.G.number_of_nodes() * 100),
                'clustering_change': ((high_removed_metrics['clustering_coefficient'] - original_metrics['avg_clustering'])
                                    / original_metrics['avg_clustering'] * 100)
            },
            'random_removal': {
                'path_length_change': ((random_removed_metrics['avg_path_length'] - original_metrics['avg_path_length'])
                                     / original_metrics['avg_path_length'] * 100),
                'num_components': random_removed_metrics['num_components'],
                'largest_component_change': ((random_removed_metrics['largest_component_size'] - self.G.number_of_nodes())
                                           / self.G.number_of_nodes() * 100),
                'clustering_change': ((random_removed_metrics['clustering_coefficient'] - original_metrics['avg_clustering'])
                                    / original_metrics['avg_clustering'] * 100)
            }
        }
        
        # Get characteristics of top 5 betweenness nodes
        top_5_nodes = []
        for node, betweenness in top_nodes[:5]:
            top_5_nodes.append({
                'node_id': node,
                'betweenness': betweenness,
                'degree': self.G.degree(node)
            })
        
        results['top_5_nodes'] = top_5_nodes
        
        return results

def main():
    """Main function to run network analysis from command line."""
    parser = argparse.ArgumentParser(description='Analyze complex networks')
    parser.add_argument('input_file', type=str, help='Path to input network file')
    parser.add_argument('--output', type=str, help='Output directory for results', default='results')
    parser.add_argument('--removal-percentage', type=float, default=0.01,
                       help='Percentage of high betweenness nodes to remove (default: 0.01 for 1%)')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Initialize and run analysis
        analyzer = NetworkAnalyzer.from_file(args.input_file)
        
        # Basic analysis
        metrics = analyzer.basic_analysis()
        logger.info("Basic network metrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value}")
        
        # Degree distribution
        degree_df = analyzer.analyze_degree_distribution()
        degree_df.to_csv(output_dir / 'degree_distribution.csv', index=False)
        
        # Community detection
        communities = analyzer.detect_communities()
        logger.info(f"Detected {len(communities)} communities")
        
        # Analyze impact of removing top 1% betweenness nodes
        removal_results = analyzer.analyze_high_betweenness_removal(percentage=0.01)
        
        # Save removal analysis results
        with open(output_dir / 'removal_analysis.txt', 'w') as f:
            f.write("Impact of High Betweenness Node Removal (top 1%):\n")
            f.write("-" * 50 + "\n")
            for metric, value in removal_results['high_betweenness_removal'].items():
                f.write(f"{metric}: {value:.2f}%\n")
            
            f.write("\nImpact of Random Node Removal:\n")
            f.write("-" * 50 + "\n")
            for metric, value in removal_results['random_removal'].items():
                f.write(f"{metric}: {value:.2f}%\n")
            
            f.write("\nTop 5 Nodes by Betweenness Centrality:\n")
            f.write("-" * 50 + "\n")
            for node in removal_results['top_5_nodes']:
                f.write(f"Node {node['node_id']}: Betweenness={node['betweenness']:.4f}, "
                       f"Degree={node['degree']}\n")
        
        # Visualize
        analyzer.visualize_network(output_dir / 'network_visualization.png')
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()