# Hidden Bridge Detection in Complex Networks

This repository contains tools for identifying and analyzing hidden bridge nodes in complex networks. Hidden bridges are nodes with low betweenness centrality that significantly impact connectivity between specific communities when removed.

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

## Usage

The project consists of three main scripts:

1. `network_analysis.py`: Basic network analysis and visualization
```python
python network_analysis.py --input path/to/network.txt
```

2. `analyze_bridges.py`: Analyze bridge node characteristics
```python
python analyze_bridges.py --input path/to/network.txt
```

3. `bridge_detection.py`: Detect and analyze hidden bridges
```python
python bridge_detection.py --input path/to/network.txt
```

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