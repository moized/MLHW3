"""
Visualization Utilities Module
Provides functions for plotting histograms, elbow curves, silhouette plots, and PCA visualizations.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from typing import List, Optional
import os


# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def ensure_results_dir(path: str = "results") -> str:
    """
    Ensure results directory exists.
    
    Args:
        path: Path to results directory
        
    Returns:
        Path to results directory
    """
    os.makedirs(path, exist_ok=True)
    return path


def plot_rfm_histograms(rfm_df: pd.DataFrame, title_suffix: str = "", 
                        save_path: Optional[str] = None) -> None:
    """
    Plot histograms for RFM features.
    
    Args:
        rfm_df: DataFrame with RFM values
        title_suffix: Suffix to add to plot title (e.g., "(Before Transform)")
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    features = ['Recency', 'Frequency', 'Monetary']
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    for ax, feature, color in zip(axes, features, colors):
        ax.hist(rfm_df[feature], bins=50, color=color, edgecolor='white', alpha=0.7)
        ax.set_xlabel(feature, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{feature} Distribution {title_suffix}', fontsize=14)
        
        # Add statistics text
        mean_val = rfm_df[feature].mean()
        median_val = rfm_df[feature].median()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='orange', linestyle='-.', linewidth=2, label=f'Median: {median_val:.2f}')
        ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        ensure_results_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_elbow_curve(k_values: List[int], inertias: List[float], 
                     optimal_k: Optional[int] = None,
                     save_path: Optional[str] = None) -> None:
    """
    Plot Elbow Method curve.
    
    Args:
        k_values: List of k values
        inertias: Corresponding inertia (WCSS) values
        optimal_k: Optimal k value to highlight
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(k_values, inertias, 'b-o', linewidth=2, markersize=8, label='Inertia (WCSS)')
    
    if optimal_k is not None:
        idx = k_values.index(optimal_k)
        ax.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2, 
                   label=f'Optimal k = {optimal_k}')
        ax.scatter([optimal_k], [inertias[idx]], color='red', s=200, zorder=5, 
                   edgecolors='black', linewidth=2)
    
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
    ax.set_title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
    ax.set_xticks(k_values)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        ensure_results_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_silhouette_curve(k_values: List[int], silhouette_scores: List[float],
                          optimal_k: Optional[int] = None,
                          save_path: Optional[str] = None) -> None:
    """
    Plot Silhouette Analysis curve.
    
    Args:
        k_values: List of k values
        silhouette_scores: Corresponding silhouette scores
        optimal_k: Optimal k value to highlight
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(k_values, silhouette_scores, 'g-o', linewidth=2, markersize=8, 
            label='Silhouette Score')
    
    if optimal_k is not None:
        idx = k_values.index(optimal_k)
        ax.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2,
                   label=f'Optimal k = {optimal_k}')
        ax.scatter([optimal_k], [silhouette_scores[idx]], color='red', s=200, 
                   zorder=5, edgecolors='black', linewidth=2)
    
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title('Silhouette Analysis for Optimal k', fontsize=14, fontweight='bold')
    ax.set_xticks(k_values)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add reference line at 0
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        ensure_results_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_combined_analysis(k_values: List[int], inertias: List[float], 
                           silhouette_scores: List[float],
                           elbow_k: int, silhouette_k: int,
                           save_path: Optional[str] = None) -> None:
    """
    Plot combined Elbow and Silhouette analysis.
    
    Args:
        k_values: List of k values
        inertias: Inertia values
        silhouette_scores: Silhouette scores
        elbow_k: Optimal k from elbow method
        silhouette_k: Optimal k from silhouette analysis
        save_path: Path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Elbow plot
    ax1.plot(k_values, inertias, 'b-o', linewidth=2, markersize=8)
    ax1.axvline(x=elbow_k, color='red', linestyle='--', linewidth=2, 
                label=f'Optimal k = {elbow_k}')
    ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax1.set_ylabel('Inertia (WCSS)', fontsize=12)
    ax1.set_title('Elbow Method', fontsize=14, fontweight='bold')
    ax1.set_xticks(k_values)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Silhouette plot
    ax2.plot(k_values, silhouette_scores, 'g-o', linewidth=2, markersize=8)
    ax2.axvline(x=silhouette_k, color='red', linestyle='--', linewidth=2,
                label=f'Optimal k = {silhouette_k}')
    ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax2.set_ylabel('Silhouette Score', fontsize=12)
    ax2.set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
    ax2.set_xticks(k_values)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        ensure_results_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def apply_pca(X: np.ndarray, n_components: int = 2) -> tuple:
    """
    Apply PCA for dimensionality reduction.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        n_components: Number of components to keep
        
    Returns:
        Tuple of (transformed data, fitted PCA model)
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    print(f"\nPCA Explained Variance Ratio:")
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        print(f"  PC{i+1}: {ratio:.4f} ({ratio*100:.2f}%)")
    print(f"  Total: {sum(pca.explained_variance_ratio_):.4f} ({sum(pca.explained_variance_ratio_)*100:.2f}%)")
    
    return X_pca, pca


def plot_pca_clusters(X_pca: np.ndarray, labels: np.ndarray, 
                      cluster_centers_pca: Optional[np.ndarray] = None,
                      save_path: Optional[str] = None) -> None:
    """
    Plot clusters in PCA-reduced 2D space.
    
    Args:
        X_pca: PCA-transformed features (n_samples, 2)
        labels: Cluster labels
        cluster_centers_pca: PCA-transformed cluster centers
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get unique clusters
    unique_clusters = np.unique(labels)
    n_clusters = len(unique_clusters)
    
    # Color palette
    colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))
    
    # Plot each cluster
    for cluster, color in zip(unique_clusters, colors):
        mask = labels == cluster
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=[color], label=f'Cluster {cluster}',
                   alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
    
    # Plot cluster centers
    if cluster_centers_pca is not None:
        ax.scatter(cluster_centers_pca[:, 0], cluster_centers_pca[:, 1],
                   c='black', marker='X', s=300, edgecolors='white', 
                   linewidth=2, label='Centroids', zorder=10)
    
    ax.set_xlabel('Principal Component 1', fontsize=12)
    ax.set_ylabel('Principal Component 2', fontsize=12)
    ax.set_title('Customer Segments (PCA Visualization)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        ensure_results_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_cluster_characteristics(interpretation_df: pd.DataFrame,
                                 save_path: Optional[str] = None) -> None:
    """
    Plot radar/bar chart showing cluster characteristics.
    
    Args:
        interpretation_df: DataFrame with cluster interpretations
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    n_clusters = len(interpretation_df)
    colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))
    
    metrics = ['Recency_Mean', 'Frequency_Mean', 'Monetary_Mean']
    titles = ['Recency (days)', 'Frequency (orders)', 'Monetary ($)']
    
    for ax, metric, title in zip(axes, metrics, titles):
        bars = ax.bar(interpretation_df['Cluster'], interpretation_df[metric], 
                      color=colors, edgecolor='black', linewidth=1)
        ax.set_xlabel('Cluster', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(f'{title} by Cluster', fontsize=14, fontweight='bold')
        ax.set_xticks(interpretation_df['Cluster'])
        
        # Add value labels on bars
        for bar, val in zip(bars, interpretation_df[metric]):
            height = bar.get_height()
            ax.annotate(f'{val:,.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        ensure_results_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def plot_3d_clusters(X: np.ndarray, labels: np.ndarray,
                     feature_names: List[str] = ['Recency', 'Frequency', 'Monetary'],
                     save_path: Optional[str] = None) -> None:
    """
    Plot clusters in 3D space (original features).
    
    Args:
        X: Feature matrix (n_samples, 3)
        labels: Cluster labels
        feature_names: Names of the 3 features
        save_path: Path to save the figure
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    unique_clusters = np.unique(labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_clusters)))
    
    for cluster, color in zip(unique_clusters, colors):
        mask = labels == cluster
        ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2],
                   c=[color], label=f'Cluster {cluster}',
                   alpha=0.6, s=30)
    
    ax.set_xlabel(feature_names[0], fontsize=10)
    ax.set_ylabel(feature_names[1], fontsize=10)
    ax.set_zlabel(feature_names[2], fontsize=10)
    ax.set_title('3D Customer Segments', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        ensure_results_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


if __name__ == "__main__":
    # Test visualizations with sample data
    print("Testing visualization module...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    sample_rfm = pd.DataFrame({
        'CustomerID': range(1, n_samples + 1),
        'Recency': np.random.exponential(50, n_samples),
        'Frequency': np.random.poisson(5, n_samples) + 1,
        'Monetary': np.random.exponential(500, n_samples)
    })
    
    # Test histogram
    plot_rfm_histograms(sample_rfm, "(Test)", "results/test_histogram.png")
    print("Histogram test complete!")
