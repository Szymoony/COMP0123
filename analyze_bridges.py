#!/usr/bin/env python3
"""
Bridge Analysis Module

This module provides functionality for analyzing bridge nodes in networks,
including their structural properties and impact on network connectivity.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BridgeAnalyzer:
    """Class for analyzing bridge nodes in networks."""
    
    def __init__(self, bridge_data: pd.DataFrame):
        """
        Initialize BridgeAnalyzer with bridge node data.
        
        Args:
            bridge_data: DataFrame containing bridge node information
        """
        self.data = bridge_data
        self._standardize_columns()
        
    @classmethod
    def from_csv(cls, filepath: str) -> 'BridgeAnalyzer':
        """
        Create BridgeAnalyzer instance from a CSV file.
        
        Args:
            filepath: Path to CSV file containing bridge data
            
        Returns:
            BridgeAnalyzer instance
        """
        try:
            df = pd.read_csv(filepath, sep=None, engine='python')
            return cls(df)
        except Exception as e:
            logger.error(f"Error loading bridge data: {str(e)}")
            raise
            
    def _standardize_columns(self):
        """Standardize column names in the dataset."""
        column_mapping = {
            'Betweenness Centrality': 'BetweennessCentrality',
            'Impact on Path Length': 'ImpactonPathLength',
            'Community Pair': 'CommunityPair',
            'Clustering Coefficient': 'ClusteringCoefficient'
        }
        self.data = self.data.rename(columns=column_mapping)
        
        # Convert numeric columns
        numeric_columns = ['BetweennessCentrality', 'ImpactonPathLength', 
                         'Degree', 'ClusteringCoefficient']
        for col in numeric_columns:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        
        # Handle infinity values
        self.data = self.data.replace('inf', np.inf)

    def categorize_impact(self) -> pd.DataFrame:
        """
        Categorize nodes by their impact level and calculate statistics.
        
        Returns:
            DataFrame containing impact statistics by category
        """
        # Create impact categories (excluding infinite values)
        finite_df = self.data[self.data['ImpactonPathLength'] != np.inf].copy()
        
        conditions = [
            (finite_df['ImpactonPathLength'] > 0.50),
            (finite_df['ImpactonPathLength'] > 0.20) & (finite_df['ImpactonPathLength'] <= 0.50),
            (finite_df['ImpactonPathLength'] > 0.05) & (finite_df['ImpactonPathLength'] <= 0.20),
            (finite_df['ImpactonPathLength'] > 0.0001) & (finite_df['ImpactonPathLength'] <= 0.05)
        ]
        choices = ['Critical', 'High', 'Moderate', 'Low']
        finite_df['ImpactCategory'] = np.select(conditions, choices, default='Other')
        
        return self._calculate_category_stats(finite_df)

    def _calculate_category_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate statistics for each impact category.
        
        Args:
            df: DataFrame with categorized impacts
            
        Returns:
            DataFrame containing statistics by category
        """
        stats = []
        for category in ['Critical', 'High', 'Moderate', 'Low']:
            cat_df = df[df['ImpactCategory'] == category]
            if not cat_df.empty:
                stats.append({
                    'Impact Category': category,
                    'Count': len(cat_df),
                    'Median Betweenness': cat_df['BetweennessCentrality'].median(),
                    'Median Degree': cat_df['Degree'].median(),
                    'Median Clustering': cat_df['ClusteringCoefficient'].median()
                })
        
        # Add infinite impact cases
        inf_cases = self.data[self.data['ImpactonPathLength'] == np.inf]
        if len(inf_cases) > 0:
            stats.append({
                'Impact Category': 'Infinite',
                'Count': len(inf_cases),
                'Median Betweenness': inf_cases['BetweennessCentrality'].median(),
                'Median Degree': inf_cases['Degree'].median(),
                'Median Clustering': inf_cases['ClusteringCoefficient'].median()
            })
        
        return pd.DataFrame(stats)

    def visualize_distributions(self, output_dir: Path):
        """
        Create visualizations of bridge node characteristics.
        
        Args:
            output_dir: Directory to save visualizations
        """
        # Set up the figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Betweenness centrality distribution
        sns.histplot(data=self.data, x='BetweennessCentrality', 
                    kde=True, ax=axes[0,0])
        axes[0,0].set_title('Betweenness Centrality Distribution')
        axes[0,0].set_xscale('log')
        
        # Impact distribution (excluding inf)
        finite_impact = self.data[self.data['ImpactonPathLength'] != np.inf]
        sns.histplot(data=finite_impact, x='ImpactonPathLength', 
                    kde=True, ax=axes[0,1])
        axes[0,1].set_title('Impact Distribution (Excluding Inf)')
        
        # Degree distribution
        sns.histplot(data=self.data, x='Degree', kde=True, ax=axes[1,0])
        axes[1,0].set_title('Degree Distribution')
        axes[1,0].set_xscale('log')
        
        # Clustering coefficient distribution
        sns.histplot(data=self.data, x='ClusteringCoefficient', 
                    kde=True, ax=axes[1,1])
        axes[1,1].set_title('Clustering Coefficient Distribution')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'bridge_distributions.png')
        plt.close()

    def generate_latex_tables(self) -> str:
        """
        Generate LaTeX tables for the analysis results.
        
        Returns:
            String containing LaTeX table code
        """
        impact_stats = self.categorize_impact()
        
        latex_table = "\\begin{table}[htbp]\n"
        latex_table += "    \\centering\n"
        latex_table += "    \\caption{Impact Distribution of Hidden Bridge Nodes}\n"
        latex_table += "    \\label{tab:impact-distribution}\n"
        latex_table += "    \\begin{tabular}{@{}lrrrr@{}}\n"
        latex_table += "    \\toprule\n"
        latex_table += "    Impact Category & Count & Median Betweenness & "
        latex_table += "Median Degree & Median Clustering \\\\\n"
        latex_table += "    \\midrule\n"
        
        for _, row in impact_stats.iterrows():
            latex_table += f"    {row['Impact Category']} & {row['Count']} & "
            latex_table += f"{row['Median Betweenness']:.4f} & "
            latex_table += f"{row['Median Degree']:.0f} & "
            latex_table += f"{row['Median Clustering']:.4f} \\\\\n"
        
        latex_table += "    \\bottomrule\n"
        latex_table += "    \\end{tabular}\n"
        latex_table += "\\end{table}"
        
        return latex_table

def main():
    """Main function to run bridge analysis from command line."""
    parser = argparse.ArgumentParser(description='Analyze bridge nodes')
    parser.add_argument('input_file', type=str, 
                       help='Path to CSV file containing bridge data')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory for results')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Initialize analyzer and run analysis
        analyzer = BridgeAnalyzer.from_csv(args.input_file)
        
        # Generate impact statistics
        impact_stats = analyzer.categorize_impact()
        impact_stats.to_csv(output_dir / 'impact_statistics.csv', index=False)
        logger.info("Generated impact statistics")
        
        # Create visualizations
        analyzer.visualize_distributions(output_dir)
        logger.info("Generated visualizations")
        
        # Generate LaTeX tables
        latex_content = analyzer.generate_latex_tables()
        with open(output_dir / 'tables.tex', 'w') as f:
            f.write(latex_content)
        logger.info("Generated LaTeX tables")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
    
# import pandas as pd
# import numpy as np

# def categorize_impact(df):
#     """Categorize nodes by their impact level and calculate statistics"""
#     # Create impact categories (excluding infinite values)
#     finite_df = df[df['ImpactonPathLength'] != np.inf].copy()
    
#     # Define impact categories
#     conditions = [
#         (finite_df['ImpactonPathLength'] > 0.50),
#         (finite_df['ImpactonPathLength'] > 0.20) & (finite_df['ImpactonPathLength'] <= 0.50),
#         (finite_df['ImpactonPathLength'] > 0.05) & (finite_df['ImpactonPathLength'] <= 0.20),
#         (finite_df['ImpactonPathLength'] > 0.0001) & (finite_df['ImpactonPathLength'] <= 0.05)
#     ]
#     choices = ['Critical', 'High', 'Moderate', 'Low']
#     finite_df['ImpactCategory'] = np.select(conditions, choices, default='Other')
    
#     # Calculate statistics for each category
#     stats = []
#     for category in ['Critical', 'High', 'Moderate', 'Low']:
#         cat_df = finite_df[finite_df['ImpactCategory'] == category]
#         if not cat_df.empty:
#             stats.append({
#                 'Impact Category': category,
#                 'Count': len(cat_df),
#                 'Median Betweenness': cat_df['BetweennessCentrality'].median(),
#                 'Median Degree': cat_df['Degree'].median(),
#                 'Median Clustering': cat_df['ClusteringCoefficient'].median()
#             })
    
#     # Add infinite impact cases as a separate category
#     inf_cases = df[df['ImpactonPathLength'] == np.inf]
#     if len(inf_cases) > 0:
#         stats.append({
#             'Impact Category': 'Infinite',
#             'Count': len(inf_cases),
#             'Median Betweenness': inf_cases['BetweennessCentrality'].median(),
#             'Median Degree': inf_cases['Degree'].median(),
#             'Median Clustering': inf_cases['ClusteringCoefficient'].median()
#         })
    
#     return pd.DataFrame(stats)

# def analyze_networks():
#     networks = {
#         'Facebook': 'hidden_bridges_facebook.csv',
#         'SBM': 'hidden_bridges_facebook_random.csv',
#         'Twitch': 'hidden_bridges_twitch_ru_100.csv'
#     }
    
#     for network_name, filepath in networks.items():
#         print(f"\n{network_name} Network Impact Distribution:")
#         print("-" * 80)
        
#         try:
#             # Load and clean data
#             df = pd.read_csv(filepath, sep=None, engine='python')
            
#             # Standardize column names
#             column_mapping = {
#                 'Betweenness Centrality': 'BetweennessCentrality',
#                 'Impact on Path Length': 'ImpactonPathLength',
#                 'Community Pair': 'CommunityPair',
#                 'Clustering Coefficient': 'ClusteringCoefficient'
#             }
#             df = df.rename(columns=column_mapping)
            
#             # Convert numeric columns
#             numeric_columns = ['BetweennessCentrality', 'ImpactonPathLength', 'Degree', 'ClusteringCoefficient']
#             for col in numeric_columns:
#                 df[col] = pd.to_numeric(df[col], errors='coerce')
            
#             # Replace 'inf' strings with np.inf
#             df = df.replace('inf', np.inf)
            
#             # Calculate impact distribution
#             impact_stats = categorize_impact(df)
            
#             # Print LaTeX table
#             print("LaTeX Table Format:")
#             print("\\begin{table}[htbp]")
#             print("    \\centering")
#             print(f"    \\caption{{Impact Distribution of Hidden Bridge Nodes in {network_name} Network}}")
#             print(f"    \\label{{tab:impact-distribution-{network_name.lower()}}}")
#             print("    \\begin{tabular}{@{}lrrrr@{}}")
#             print("    \\toprule")
#             print("    Impact Category & Count & Median Betweenness & Median Degree & Median Clustering \\\\")
#             print("    \\midrule")
            
#             for _, row in impact_stats.iterrows():
#                 print(f"    {row['Impact Category']} & {row['Count']} & {row['Median Betweenness']:.4f} & {row['Median Degree']:.0f} & {row['Median Clustering']:.4f} \\\\")
            
#             print("    \\bottomrule")
#             print("    \\end{tabular}")
#             print("\\end{table}")
            
#             # Print summary statistics
#             print("\nSummary Statistics:")
#             print(impact_stats.to_string(index=False))
            
#         except Exception as e:
#             print(f"Error processing {network_name} network: {str(e)}")

# if __name__ == "__main__":
#     analyze_networks()










# # import pandas as pd
# # import seaborn as sns
# # import matplotlib.pyplot as plt
# # import numpy as np
# # from scipy import stats
# # pd.options.mode.chained_assignment = None  # Disable copy warnings

# # def load_and_clean_data(filepath):
# #     """Load and clean the CSV data"""
# #     try:
# #         # Try reading with comma delimiter first
# #         try:
# #             df = pd.read_csv(filepath)
# #         except:
# #             # If that fails, try semicolon delimiter
# #             df = pd.read_csv(filepath, sep=';')
        
# #         print(f"Loaded columns: {df.columns.tolist()}")
        
# #         # If we have a single column with semicolons, split it
# #         if len(df.columns) == 1:
# #             df = pd.DataFrame([x.split(';') for x in df[df.columns[0]].values.tolist()],
# #                             columns=['Node', 'Betweenness Centrality', 'Impact on Path Length', 
# #                                    'Community Pair', 'Degree', 'Clustering Coefficient'])
        
# #         # Standardize column names
# #         column_mapping = {
# #             'Betweenness Centrality': 'BetweennessCentrality',
# #             'Impact on Path Length': 'ImpactonPathLength',
# #             'Community Pair': 'CommunityPair',
# #             'Clustering Coefficient': 'ClusteringCoefficient'
# #         }
# #         df = df.rename(columns=column_mapping)
        
# #         # Convert numeric columns
# #         numeric_columns = ['BetweennessCentrality', 'ImpactonPathLength', 'Degree', 'ClusteringCoefficient']
# #         for col in numeric_columns:
# #             df[col] = pd.to_numeric(df[col], errors='coerce')
        
# #         # Replace 'inf' with np.inf
# #         df = df.replace('inf', np.inf)
        
# #         print(f"Final columns: {df.columns.tolist()}")
# #         return df
    
# #     except Exception as e:
# #         print(f"Error loading data from {filepath}: {str(e)}")
# #         raise

# # def analyze_bridge_distributions(df, network_name):
# #     """Analyze distributions of key metrics for bridge nodes"""
# #     try:
# #         # Set up the figure
# #         fig, axes = plt.subplots(2, 2, figsize=(15, 10))
# #         fig.suptitle(f'Bridge Node Characteristics - {network_name}', fontsize=16)
        
# #         # Plot distributions
# #         sns.histplot(data=df, x='BetweennessCentrality', kde=True, ax=axes[0,0])
# #         axes[0,0].set_title('Betweenness Centrality Distribution')
# #         axes[0,0].set_xscale('log')
        
# #         # Filter out inf values for path length impact visualization
# #         finite_impact = df[df['ImpactonPathLength'] != np.inf].copy()
# #         sns.histplot(data=finite_impact, x='ImpactonPathLength', kde=True, ax=axes[0,1])
# #         axes[0,1].set_title('Impact Distribution (Excluding Inf)')
        
# #         sns.histplot(data=df, x='Degree', kde=True, ax=axes[1,0])
# #         axes[1,0].set_title('Degree Distribution')
# #         axes[1,0].set_xscale('log')
        
# #         sns.histplot(data=df, x='ClusteringCoefficient', kde=True, ax=axes[1,1])
# #         axes[1,1].set_title('Clustering Coefficient Distribution')
        
# #         plt.tight_layout()
# #         return fig
# #     except Exception as e:
# #         print(f"Error in analyze_bridge_distributions: {str(e)}")
# #         raise

# # def analyze_community_impact(df):
# #     """Analyze which community pairs are most affected by bridge nodes"""
# #     try:
# #         # Count frequency of community pairs
# #         pair_counts = df['CommunityPair'].value_counts().head(10)
        
# #         plt.figure(figsize=(12, 6))
# #         pair_counts.plot(kind='bar')
# #         plt.title('Most Frequently Affected Community Pairs')
# #         plt.xlabel('Community Pair')
# #         plt.ylabel('Number of Bridge Nodes')
# #         plt.xticks(rotation=45)
# #         plt.tight_layout()
# #         return plt.gcf()
# #     except Exception as e:
# #         print(f"Error in analyze_community_impact: {str(e)}")
# #         raise

# # def analyze_structural_properties(df):
# #     """Analyze and visualize structural properties of hidden bridge nodes"""
# #     try:
# #         fig, axes = plt.subplots(2, 2, figsize=(15, 12))
# #         fig.suptitle('Structural Properties of Hidden Bridge Nodes', fontsize=16)
        
# #         # Create a copy of the DataFrame for finite impact values
# #         finite_impact = df[df['ImpactonPathLength'] != np.inf].copy()
        
# #         # Betweenness vs Impact
# #         sns.scatterplot(
# #             data=finite_impact,
# #             x='BetweennessCentrality',
# #             y='ImpactonPathLength',
# #             ax=axes[0,0],
# #             alpha=0.5
# #         )
# #         axes[0,0].set_xscale('log')
# #         axes[0,0].set_title('Betweenness Centrality vs Impact')
        
# #         # Degree vs Clustering Coefficient
# #         sns.scatterplot(
# #             data=df,
# #             x='Degree',
# #             y='ClusteringCoefficient',
# #             ax=axes[0,1],
# #             alpha=0.5
# #         )
# #         axes[0,1].set_xscale('log')
# #         axes[0,1].set_title('Degree vs Clustering Coefficient')
        
# #         # Impact distribution by degree quartile
# #         finite_impact.loc[:, 'DegreeQuartile'] = pd.qcut(finite_impact['Degree'], 
# #                                                         q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
# #         sns.boxplot(
# #             data=finite_impact,
# #             x='DegreeQuartile',
# #             y='ImpactonPathLength',
# #             ax=axes[1,0]
# #         )
# #         axes[1,0].set_title('Impact Distribution by Degree Quartile')
        
# #         # Clustering coefficient distribution by impact severity
# #         finite_impact.loc[:, 'ImpactCategory'] = pd.qcut(
# #             finite_impact['ImpactonPathLength'],
# #             q=4,
# #             labels=['Low', 'Medium-Low', 'Medium-High', 'High']
# #         )
# #         sns.boxplot(
# #             data=finite_impact,
# #             x='ImpactCategory',
# #             y='ClusteringCoefficient',
# #             ax=axes[1,1]
# #         )
# #         axes[1,1].set_title('Clustering Coefficient by Impact Category')
        
# #         plt.tight_layout()
# #         return fig
# #     except Exception as e:
# #         print(f"Error in analyze_structural_properties: {str(e)}")
# #         raise

# # def analyze_property_correlations(df):
# #     """Calculate correlations between structural properties"""
# #     try:
# #         # Filter out infinite values for correlation analysis
# #         finite_df = df[df['ImpactonPathLength'] != np.inf]
        
# #         # Calculate correlations
# #         corr_matrix = finite_df[[
# #             'BetweennessCentrality',
# #             'ImpactonPathLength',
# #             'Degree',
# #             'ClusteringCoefficient'
# #         ]].corr()
        
# #         # Create correlation heatmap
# #         plt.figure(figsize=(10, 8))
# #         sns.heatmap(
# #             corr_matrix,
# #             annot=True,
# #             cmap='coolwarm',
# #             center=0,
# #             vmin=-1,
# #             vmax=1,
# #             fmt='.2f'
# #         )
# #         plt.title('Correlations Between Structural Properties')
        
# #         return plt.gcf(), corr_matrix
# #     except Exception as e:
# #         print(f"Error in analyze_property_correlations: {str(e)}")
# #         raise

# # def generate_structural_statistics(df):
# #     """Generate detailed statistics about structural properties"""
# #     try:
# #         # Calculate quartiles for each metric
# #         property_quartiles = {
# #             'BetweennessCentrality': df['BetweennessCentrality'].quantile([0.25, 0.5, 0.75]).to_dict(),
# #             'Degree': df['Degree'].quantile([0.25, 0.5, 0.75]).to_dict(),
# #             'ClusteringCoefficient': df['ClusteringCoefficient'].quantile([0.25, 0.5, 0.75]).to_dict()
# #         }
        
# #         # Calculate statistics for nodes with infinite impact
# #         inf_impact_nodes = df[df['ImpactonPathLength'] == np.inf]
# #         inf_impact_stats = {
# #             'Count': len(inf_impact_nodes),
# #             'Median Betweenness': inf_impact_nodes['BetweennessCentrality'].median(),
# #             'Median Degree': inf_impact_nodes['Degree'].median(),
# #             'Median Clustering': inf_impact_nodes['ClusteringCoefficient'].median()
# #         }
        
# #         # Statistical tests
# #         finite_impact = df[df['ImpactonPathLength'] != np.inf]
# #         degree_impact_corr, degree_impact_p = stats.spearmanr(
# #             finite_impact['Degree'],
# #             finite_impact['ImpactonPathLength']
# #         )
        
# #         statistical_tests = {
# #             'Degree-Impact Correlation': degree_impact_corr,
# #             'Degree-Impact P-value': degree_impact_p
# #         }
        
# #         return {
# #             'Property Quartiles': property_quartiles,
# #             'Infinite Impact Stats': inf_impact_stats,
# #             'Statistical Tests': statistical_tests
# #         }
# #     except Exception as e:
# #         print(f"Error in generate_structural_statistics: {str(e)}")
# #         raise

# # def main():
# #     networks = {
# #         'Facebook': 'hidden_bridges_facebook.csv',
# #         'SBM': 'hidden_bridges_facebook_random.csv',
# #         'Twitch': 'hidden_bridges_twitch_ru_100.csv'
# #     }
    
# #     results = {}
# #     for network_name, filepath in networks.items():
# #         print(f"\nProcessing {network_name} network...")
# #         try:
# #             df = load_and_clean_data(filepath)
            
# #             # Generate and save distributions plot
# #             fig = analyze_bridge_distributions(df, network_name)
# #             fig.savefig(f'{network_name.lower()}_distributions.png')
# #             plt.close(fig)
            
# #             # Generate and save community impact plot
# #             fig = analyze_community_impact(df)
# #             fig.savefig(f'{network_name.lower()}_community_impact.png')
# #             plt.close(fig)
            
# #             # Generate and save structural properties plot
# #             fig = analyze_structural_properties(df)
# #             fig.savefig(f'{network_name.lower()}_structural_properties.png')
# #             plt.close(fig)
            
# #             # Generate summary statistics
# #             results[network_name] = {
# #                 'Basic Stats': generate_summary_statistics(df),
# #                 'Structural Stats': generate_structural_statistics(df)
# #             }
            
# #             # Generate and save correlation plot
# #             fig, corr_matrix = analyze_property_correlations(df)
# #             fig.savefig(f'{network_name.lower()}_correlations.png')
# #             plt.close(fig)
# #             results[network_name]['Correlation Matrix'] = corr_matrix
            
# #         except Exception as e:
# #             print(f"Error processing {network_name} network: {str(e)}")
# #             continue
    
# #     return results
# # def generate_summary_statistics(df):
# #     """Generate summary statistics for bridge nodes"""
# #     try:
# #         stats_dict = {
# #             'Total Bridge Nodes': len(df),
# #             'Unique Community Pairs': df['CommunityPair'].nunique(),
# #             'Average Betweenness': df['BetweennessCentrality'].mean(),
# #             'Median Betweenness': df['BetweennessCentrality'].median(),
# #             'Average Degree': df['Degree'].mean(),
# #             'Median Degree': df['Degree'].median(),
# #             'Average Clustering Coef': df['ClusteringCoefficient'].mean(),
# #             'Infinite Impact Count': sum(df['ImpactonPathLength'] == np.inf)
# #         }
# #         return stats_dict
# #     except Exception as e:
# #         print(f"Error in generate_summary_statistics: {str(e)}")
# #         raise

# # if __name__ == "__main__":
# #     results = main()
    
# #     # Print summary statistics for each network
# #     for network, stats in results.items():
# #         print(f"\n{network} Network Summary:")
# #         print("\nBasic Statistics:")
# #         for metric, value in stats['Basic Stats'].items():
# #             print(f"{metric}: {value:.4f}" if isinstance(value, float) else f"{metric}: {value}")
        
# #         print("\nStructural Statistics:")
# #         for category, values in stats['Structural Stats'].items():
# #             print(f"\n{category}:")
# #             if isinstance(values, dict):
# #                 for metric, value in values.items():
# #                     print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")
# #             else:
# #                 print(f"  {values:.4f}" if isinstance(values, float) else f"  {values}")

# import pandas as pd
# import numpy as np

# def analyze_low_betweenness_nodes(filepath, network_name):
#     """Analyze nodes with betweenness centrality below 0.05"""
#     # Load data
#     df = pd.read_csv(filepath, sep=None, engine='python')
    
#     # Standardize column names
#     column_mapping = {
#         'Betweenness Centrality': 'BetweennessCentrality',
#         'Impact on Path Length': 'ImpactonPathLength',
#         'Community Pair': 'CommunityPair',
#         'Clustering Coefficient': 'ClusteringCoefficient'
#     }
#     df = df.rename(columns=column_mapping)
    
#     # Convert to numeric and handle infinities
#     df['BetweennessCentrality'] = pd.to_numeric(df['BetweennessCentrality'], errors='coerce')
#     df['ImpactonPathLength'] = pd.to_numeric(df['ImpactonPathLength'].replace('inf', np.inf), errors='coerce')
    
#     # Identify low betweenness nodes
#     low_betweenness = df[df['BetweennessCentrality'] < 0.05]
    
#     # Calculate statistics
#     total_nodes = len(df)
#     low_betweenness_count = len(low_betweenness)
#     percentage = (low_betweenness_count / total_nodes) * 100
    
#     # Calculate impact statistics for low betweenness nodes
#     infinite_impact = sum(low_betweenness['ImpactonPathLength'] == np.inf)
#     high_impact = sum((low_betweenness['ImpactonPathLength'] > 0.5) & 
#                      (low_betweenness['ImpactonPathLength'] != np.inf))
    
#     print(f"\n{network_name} Network Analysis:")
#     print("-" * 50)
#     print(f"Total hidden bridge nodes: {total_nodes}")
#     print(f"Nodes with betweenness < 0.05: {low_betweenness_count} ({percentage:.1f}%)")
#     print(f"Among low betweenness nodes:")
#     print(f"- Causing infinite impact: {infinite_impact}")
#     print(f"- Causing high impact (>50%): {high_impact}")
    
#     # Detailed statistics for low betweenness nodes
#     if len(low_betweenness) > 0:
#         print("\nLow betweenness nodes statistics:")
#         print(f"Median degree: {low_betweenness['Degree'].median():.1f}")
#         print(f"Median clustering coefficient: {low_betweenness['ClusteringCoefficient'].median():.4f}")
        
#         # Impact distribution (excluding infinite)
#         finite_impact = low_betweenness[low_betweenness['ImpactonPathLength'] != np.inf]
#         if len(finite_impact) > 0:
#             print(f"Median impact (excluding infinite): {finite_impact['ImpactonPathLength'].median():.4f}")
    
#     return low_betweenness

# def main():
#     networks = {
#         'Facebook': 'hidden_bridges_facebook.csv',
#         'SBM': 'hidden_bridges_facebook_random.csv',
#         'Twitch': 'hidden_bridges_twitch_ru_100.csv'
#     }
    
#     results = {}
#     for network_name, filepath in networks.items():
#         try:
#             results[network_name] = analyze_low_betweenness_nodes(filepath, network_name)
#         except Exception as e:
#             print(f"Error processing {network_name} network: {str(e)}")
#             continue

# if __name__ == "__main__":
#     main()