# Hidden Bridge Detection in Complex Networks

This repository contains the implementation and analysis code for the paper "Hidden Bridges: Identifying Critical Low-Centrality Nodes in Community-Specific Social Network Connectivity". It provides tools for detecting and analyzing nodes that, despite having low betweenness centrality, significantly impact connectivity between specific communities when removed.

## Paper Abstract

This study investigates the overlooked role of "hidden bridge" nodes in social networks - nodes with low betweenness centrality that significantly impact connectivity between specific communities. Using three complementary networks (Facebook social circles, synthetic Stochastic Block Model, and Russian Twitch social network), we demonstrate that traditional high-betweenness centrality metrics fail to identify crucial nodes maintaining local community connections. Our analysis identified 466 hidden bridges in the Facebook network, with 155 nodes having betweenness centrality below 0.05, including one critical node that disconnected a single community from 15 different community pairs.

## Features

- Network loading and basic analysis
- Community detection using Louvain algorithm
- Identification of hidden bridge nodes
- Comparative analysis with synthetic networks (Stochastic Block Model)
- Visualization of network properties and communities

## Requirements

```bash
networkx>=2.8
matplotlib>=3.5
pandas>=1.4
numpy>=1.21
seaborn>=0.11
tqdm>=4.64
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Szymoony/COMP0123.git
cd COMP0123
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Reproducing Results

To reproduce the results from our paper:

1. Download the required datasets:
   - Facebook social circles dataset: [SNAP](http://snap.stanford.edu/data/egonets-Facebook.html)
   - Russian Twitch social network: [SNAP](https://snap.stanford.edu/data/twitch-social-networks.html)

2. Run the analysis pipeline:

```bash
# 1. First analyze the network structure
python network_analysis.py --input facebook_combined.txt --output results/facebook

# 2. Detect hidden bridges
python bridge_detection.py --input facebook_combined.txt --output results/facebook \
    --betweenness-threshold 0.05 --impact-threshold 0.01

# 3. Generate comparison SBM network
python bridge_detection.py --input facebook_combined.txt --output results/sbm \
    --generate-sbm --betweenness-threshold 0.05 --impact-threshold 0.01

# 4. Analyze bridge characteristics
python analyze_bridges.py --input results/facebook/hidden_bridges.csv --output results/facebook
```

Key parameters used in our analysis:
- Betweenness threshold: 0.05
- Impact threshold: 0.01 (1% increase in path length)
- Community detection: Louvain algorithm
- SBM parameters matched to maintain original network's degree distribution and community sizes

### Input Format

The scripts accept network data in edge list format:
- Text file with one edge per line
- Each line should contain two node IDs separated by whitespace
- Node IDs should be integers
- Undirected edges (order doesn't matter)

Example:
```
1 2
1 3
2 3
3 4
```

## Output

The scripts generate several outputs:
- CSV files containing identified hidden bridges
- Network visualizations
- Community structure analysis
- Statistical analysis of bridge nodes

## Example

```python
import networkx as nx
from bridge_detection import identify_hidden_bridges

# Load your network
G = nx.read_edgelist("your_network.txt", nodetype=int)

# Detect communities and hidden bridges
communities = detect_communities(G)
hidden_bridges = identify_hidden_bridges(G, communities)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:
```
@misc{COMP0123_hidden_bridges,
    author = {Goralczuk, Szymon},
    title = {Hidden Bridges: Identifying Critical Low-Centrality Nodes in Community-Specific Social Network Connectivity},
    year = {2025},
    publisher = {GitHub},
    url = {https://github.com/Szymoony/COMP0123}
}
```