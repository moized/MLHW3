"""
Evaluation Script for Customer Segmentation
Loads results and provides detailed analysis of clusters.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt


def load_results(results_dir: str = "results") -> dict:
    """
    Load saved results from previous training.
    
    Args:
        results_dir: Directory containing results
        
    Returns:
        Dictionary with loaded data
    """
    results = {}
    
    # Load customer segments
    segments_path = os.path.join(results_dir, "customer_segments.csv")
    if os.path.exists(segments_path):
        results['segments'] = pd.read_csv(segments_path)
        print(f"Loaded customer segments: {len(results['segments']):,} customers")
    
    # Load cluster interpretation
    interpretation_path = os.path.join(results_dir, "cluster_interpretation.csv")
    if os.path.exists(interpretation_path):
        results['interpretation'] = pd.read_csv(interpretation_path)
        print(f"Loaded cluster interpretation: {len(results['interpretation'])} clusters")
    
    # Load clustering analysis
    analysis_path = os.path.join(results_dir, "clustering_analysis.csv")
    if os.path.exists(analysis_path):
        results['analysis'] = pd.read_csv(analysis_path)
        print(f"Loaded clustering analysis for k = 2 to {len(results['analysis']) + 1}")
    
    return results


def evaluate_clusters(segments_df: pd.DataFrame) -> dict:
    """
    Evaluate cluster quality metrics.
    
    Args:
        segments_df: DataFrame with customer segments
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "=" * 60)
    print("CLUSTER EVALUATION METRICS")
    print("=" * 60)
    
    metrics = {}
    
    # Calculate cluster statistics
    cluster_stats = segments_df.groupby('Cluster').agg({
        'CustomerID': 'count',
        'Recency': ['mean', 'std', 'min', 'max'],
        'Frequency': ['mean', 'std', 'min', 'max'],
        'Monetary': ['mean', 'std', 'min', 'max']
    })
    
    metrics['cluster_stats'] = cluster_stats
    
    # Print cluster sizes
    print("\nCluster Sizes:")
    cluster_sizes = segments_df['Cluster'].value_counts().sort_index()
    for cluster, count in cluster_sizes.items():
        pct = count / len(segments_df) * 100
        print(f"  Cluster {cluster}: {count:,} customers ({pct:.1f}%)")
    
    metrics['cluster_sizes'] = cluster_sizes.to_dict()
    
    # RFM means by cluster
    print("\nRFM Means by Cluster:")
    rfm_means = segments_df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    print(rfm_means.round(2).to_string())
    
    metrics['rfm_means'] = rfm_means
    
    return metrics


def detailed_cluster_analysis(segments_df: pd.DataFrame) -> None:
    """
    Provide detailed analysis for each cluster.
    
    Args:
        segments_df: DataFrame with customer segments
    """
    print("\n" + "=" * 60)
    print("DETAILED CLUSTER ANALYSIS")
    print("=" * 60)
    
    # Overall statistics
    overall_means = segments_df[['Recency', 'Frequency', 'Monetary']].mean()
    
    for cluster in sorted(segments_df['Cluster'].unique()):
        cluster_data = segments_df[segments_df['Cluster'] == cluster]
        
        print(f"\n--- Cluster {cluster} ---")
        print(f"Size: {len(cluster_data):,} customers ({len(cluster_data)/len(segments_df)*100:.1f}%)")
        
        # RFM Statistics
        for feature in ['Recency', 'Frequency', 'Monetary']:
            mean_val = cluster_data[feature].mean()
            median_val = cluster_data[feature].median()
            std_val = cluster_data[feature].std()
            min_val = cluster_data[feature].min()
            max_val = cluster_data[feature].max()
            
            # Compare to overall
            vs_overall = "↑" if mean_val > overall_means[feature] else "↓"
            
            print(f"\n  {feature}:")
            print(f"    Mean: {mean_val:,.2f} {vs_overall} (Overall: {overall_means[feature]:,.2f})")
            print(f"    Median: {median_val:,.2f}")
            print(f"    Std: {std_val:,.2f}")
            print(f"    Range: [{min_val:,.2f}, {max_val:,.2f}]")
        
        # Customer segment interpretation
        r_level = "Recent" if cluster_data['Recency'].mean() < overall_means['Recency'] else "Dormant"
        f_level = "Frequent" if cluster_data['Frequency'].mean() > overall_means['Frequency'] else "Occasional"
        m_level = "High-value" if cluster_data['Monetary'].mean() > overall_means['Monetary'] else "Low-value"
        
        print(f"\n  Segment Profile: {r_level}, {f_level}, {m_level}")
        
        # Business recommendations
        print("\n  Business Recommendations:")
        if r_level == "Recent" and f_level == "Frequent" and m_level == "High-value":
            print("    → VIP customers - Priority treatment, exclusive offers")
        elif r_level == "Recent" and m_level == "High-value":
            print("    → Potential VIPs - Nurture with loyalty programs")
        elif r_level == "Dormant" and m_level == "High-value":
            print("    → At-risk high-value - Win-back campaigns needed")
        elif r_level == "Recent" and f_level == "Occasional":
            print("    → New/Promising - Encourage repeat purchases")
        elif r_level == "Dormant" and f_level == "Occasional":
            print("    → Hibernating - Reactivation campaigns")
        else:
            print("    → Monitor and analyze further")


def sample_customers(segments_df: pd.DataFrame, n_per_cluster: int = 5) -> None:
    """
    Show sample customers from each cluster.
    
    Args:
        segments_df: DataFrame with customer segments
        n_per_cluster: Number of samples per cluster
    """
    print("\n" + "=" * 60)
    print(f"SAMPLE CUSTOMERS ({n_per_cluster} per cluster)")
    print("=" * 60)
    
    for cluster in sorted(segments_df['Cluster'].unique()):
        cluster_data = segments_df[segments_df['Cluster'] == cluster]
        sample = cluster_data.sample(min(n_per_cluster, len(cluster_data)), random_state=42)
        
        print(f"\nCluster {cluster}:")
        print(sample[['CustomerID', 'Recency', 'Frequency', 'Monetary']].to_string(index=False))


def plot_cluster_comparison(segments_df: pd.DataFrame, save_path: str = None) -> None:
    """
    Create comparison plots for clusters.
    
    Args:
        segments_df: DataFrame with customer segments
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Cluster sizes pie chart
    ax1 = axes[0, 0]
    cluster_sizes = segments_df['Cluster'].value_counts().sort_index()
    colors = plt.cm.Set1(np.linspace(0, 1, len(cluster_sizes)))
    ax1.pie(cluster_sizes.values, labels=[f'Cluster {c}' for c in cluster_sizes.index],
            autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Customer Distribution by Cluster', fontsize=12, fontweight='bold')
    
    # Box plots for each RFM feature
    features = ['Recency', 'Frequency', 'Monetary']
    for idx, feature in enumerate(features):
        ax = axes[(idx + 1) // 2, (idx + 1) % 2]
        segments_df.boxplot(column=feature, by='Cluster', ax=ax)
        ax.set_title(f'{feature} by Cluster', fontsize=12, fontweight='bold')
        ax.set_xlabel('Cluster')
        ax.set_ylabel(feature)
        plt.suptitle('')  # Remove automatic title
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def main():
    """
    Main evaluation function.
    """
    print("=" * 60)
    print("CUSTOMER SEGMENTATION - EVALUATION")
    print("=" * 60)
    
    # Check if results exist
    results_dir = "results"
    if not os.path.exists(results_dir):
        print("\nError: No results found. Please run train.py first.")
        return
    
    # Load results
    results = load_results(results_dir)
    
    if 'segments' not in results:
        print("\nError: Customer segments not found. Please run train.py first.")
        return
    
    segments_df = results['segments']
    
    # Evaluate clusters
    metrics = evaluate_clusters(segments_df)
    
    # Detailed analysis
    detailed_cluster_analysis(segments_df)
    
    # Sample customers
    sample_customers(segments_df, n_per_cluster=5)
    
    # Show interpretation if available
    if 'interpretation' in results:
        print("\n" + "=" * 60)
        print("CLUSTER INTERPRETATION SUMMARY")
        print("=" * 60)
        print(results['interpretation'].to_string(index=False))
    
    # Create comparison plots
    plot_cluster_comparison(segments_df, os.path.join(results_dir, "cluster_comparison.png"))
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
