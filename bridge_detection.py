#!/usr/bin/env python3
"""
Hidden Bridge Detection Module

This module implements algorithms for detecting and analyzing hidden bridge nodes
in complex networks. Hidden bridges are nodes with low betweenness centrality
that significantly impact connectivity between specific communities.
"""

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Set, Dict, Optional, Tuple
import argparse
from tqdm import tqdm
import random
from itertools import combinations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BridgeDetector:
    """Class for detecting and analyzing hidden bridges in networks."""
    
    def __init__(self, graph: nx.Graph):
        """
        Initialize BridgeDetector with a network.
        
        Args:
            graph: NetworkX graph to analyze
        """
        self.G = graph
        self.communities = None
        self.betweenness_centrality = None
        
    @classmethod
    def from_file(cls, filepath: str) -> 'BridgeDetector':
        """
        Create BridgeDetector instance from an edge list file.
        
        Args:
            filepath: Path to edge list file
            
        Returns:
            BridgeDetector instance
        """
        try:
            G = nx.read_edgelist(filepath, nodetype=int)
            return cls(G)
        except Exception as e:
            logger.error(f"Error loading network: {str(e)}")
            raise

    def detect_communities(self) -> List[Set[int]]:
        """
        Detect communities using Louvain algorithm.
        
        Returns:
            List of sets containing nodes in each community
        """
        try:
            self.communities = list(nx.community.louvain_communities(self.G))
            logger.info(f"Detected {len(self.communities)} communities")
            return self.communities
        except Exception as e:
            logger.error(f"Community detection failed: {str(e)}")
            raise

    def calculate_betweenness(self):
        """Calculate betweenness centrality for all nodes."""
        logger.info("Calculating betweenness centrality...")
        self.betweenness_centrality = nx.betweenness_centrality(self.G)

    def identify_hidden_bridges(
        self,
        num_pairs: Optional[int] = None,
        impact_threshold: float = 0.001,
        betweenness_threshold: float = 0.05
    ) -> pd.DataFrame:
        """
        Identify hidden bridge nodes in the network.
        
        Args:
            num_pairs: Number of community pairs to analyze (None for all pairs)
            impact_threshold: Minimum impact threshold for bridge classification
            betweenness_threshold: Maximum betweenness for hidden bridge classification
            
        Returns:
            DataFrame containing identified hidden bridges and their properties
        """
        if self.communities is None:
            self.detect_communities()
        if self.betweenness_centrality is None:
            self.calculate_betweenness()

        # Generate community pairs
        community_pairs = list(combinations(range(len(self.communities)), 2))
        if num_pairs and num_pairs < len(community_pairs):
            community_pairs = random.sample(community_pairs, num_pairs)

        hidden_bridges = []
        for pair in tqdm(community_pairs, desc="Analyzing community pairs"):
            bridges = self._analyze_community_pair(pair[0], pair[1])
            hidden_bridges.extend(bridges)

        # Convert to DataFrame
        if hidden_bridges:
            df = pd.DataFrame(hidden_bridges)
            # Filter by betweenness threshold
            df = df[df['Betweenness Centrality'] <= betweenness_threshold]
            return df.sort_values('Impact on Path Length', ascending=False)
        return pd.DataFrame()

    def _analyze_community_pair(
        self,
        comm1_idx: int,
        comm2_idx: int
    ) -> List[Dict]:
        """
        Analyze potential hidden bridges between a pair of communities.
        
        Args:
            comm1_idx: Index of first community
            comm2_idx: Index of second community
            
        Returns:
            List of dictionaries containing bridge information
        """
        comm1 = self.communities[comm1_idx]
        comm2 = self.communities[comm2_idx]
        
        # Find nodes on shortest paths
        nodes_to_check = set()
        initial_paths = []
        
        for u in comm1:
            for v in comm2:
                if u != v and nx.has_path(self.G, u, v):
                    path = nx.shortest_path(self.G, u, v)
                    nodes_to_check.update(path[1:-1])
                    initial_paths.append(len(path) - 1)

        if not initial_paths:
            return []

        initial_avg_length = np.mean(initial_paths)
        bridges = []

        # Check impact of each potential bridge
        for node in nodes_to_check:
            G_temp = self.G.copy()
            G_temp.remove_node(node)
            
            new_paths = []
            for u in comm1:
                for v in comm2:
                    if (u != node and v != node and 
                        u != v and nx.has_path(G_temp, u, v)):
                        new_paths.append(nx.shortest_path_length(G_temp, u, v))

            if new_paths:
                new_avg_length = np.mean(new_paths)
                impact = (new_avg_length - initial_avg_length) / initial_avg_length
                
                bridges.append({
                    'Node': node,
                    'Betweenness Centrality': self.betweenness_centrality[node],
                    'Impact on Path Length': impact,
                    'Community Pair': f"{comm1_idx}-{comm2_idx}",
                    'Degree': self.G.degree(node),
                    'Clustering Coefficient': nx.clustering(self.G, node)
                })
            else:
                # Node disconnects the communities
                bridges.append({
                    'Node': node,
                    'Betweenness Centrality': self.betweenness_centrality[node],
                    'Impact on Path Length': float('inf'),
                    'Community Pair': f"{comm1_idx}-{comm2_idx}",
                    'Degree': self.G.degree(node),
                    'Clustering Coefficient': nx.clustering(self.G, node)
                })

        return bridges

    def generate_sbm_comparison(self) -> nx.Graph:
        """
        Generate a comparable Stochastic Block Model network.
        
        Returns:
            NetworkX graph of the SBM network
        """
        if self.communities is None:
            self.detect_communities()

        sizes = [len(c) for c in self.communities]
        
        # Calculate probabilities
        p_within = []
        p_between = []
        
        for i, comm1 in enumerate(self.communities):
            # Within-community probability
            subg = self.G.subgraph(comm1)
            edges = subg.number_of_edges()
            possible = len(comm1) * (len(comm1) - 1) / 2
            p_within.append(edges / possible if possible > 0 else 0)
            
            # Between-community probabilities
            p_between_row = []
            for j, comm2 in enumerate(self.communities[i+1:], i+1):
                edges = sum(1 for u in comm1 for v in comm2 
                          if self.G.has_edge(u, v))
                possible = len(comm1) * len(comm2)
                p_between_row.append(edges / possible if possible > 0 else 0)
            p_between.append(p_between_row)

        # Create probability matrix
        n_communities = len(self.communities)
        prob_matrix = np.zeros((n_communities, n_communities))
        
        # Fill diagonal with within-community probabilities
        np.fill_diagonal(prob_matrix, p_within)
        
        # Fill upper triangle with between-community probabilities
        for i in range(n_communities):
            for j in range(i + 1, n_communities):
                prob = p_between[i][j-i-1]
                prob_matrix[i,j] = prob
                prob_matrix[j,i] = prob

        return nx.stochastic_block_model(sizes, prob_matrix, seed=42)

def main():
    """Main function to run bridge detection from command line."""
    parser = argparse.ArgumentParser(
        description='Detect hidden bridges in complex networks')
    parser.add_argument('input_file', type=str, 
                       help='Path to network edge list file')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--pairs', type=int, default=None,
                       help='Number of community pairs to analyze')
    parser.add_argument('--generate-sbm', action='store_true',
                       help='Generate and analyze comparable SBM network')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Initialize detector
        detector = BridgeDetector.from_file(args.input_file)
        
        # Detect communities and identify bridges
        detector.detect_communities()
        bridges_df = detector.identify_hidden_bridges(num_pairs=args.pairs)
        
        # Save results
        bridges_df.to_csv(output_dir / 'hidden_bridges.csv', index=False)
        logger.info(f"Identified {len(bridges_df)} hidden bridges")
        
        if args.generate_sbm:
            # Generate and analyze SBM network
            G_sbm = detector.generate_sbm_comparison()
            sbm_detector = BridgeDetector(G_sbm)
            sbm_detector.detect_communities()
            sbm_bridges_df = sbm_detector.identify_hidden_bridges(
                num_pairs=args.pairs)
            sbm_bridges_df.to_csv(output_dir / 'sbm_hidden_bridges.csv', 
                                index=False)
            logger.info(f"Identified {len(sbm_bridges_df)} hidden bridges in SBM")
            
    except Exception as e:
        logger.error(f"Bridge detection failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()