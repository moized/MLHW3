"""
Clustering Module
Implements k-Means clustering with Elbow Method and Silhouette Analysis.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from typing import Tuple, List, Dict


def run_kmeans(X: np.ndarray, k: int, random_state: int = 42) -> KMeans:
    """
    Run k-Means clustering.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        k: Number of clusters
        random_state: Random seed for reproducibility
        
    Returns:
        Fitted KMeans model
    """
    kmeans = KMeans(
        n_clusters=k,
        init='k-means++',
        n_init=10,
        max_iter=300,
        random_state=random_state
    )
    kmeans.fit(X)
    
    return kmeans


def elbow_analysis(X: np.ndarray, k_range: range = range(2, 11)) -> Tuple[List[int], List[float]]:
    """
    Perform Elbow Method analysis to find optimal k.
    
    The Elbow Method plots the Within-Cluster Sum of Squares (WCSS/Inertia)
    for different values of k. The "elbow" point where the decrease slows
    down indicates the optimal number of clusters.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        k_range: Range of k values to test
        
    Returns:
        Tuple of (k values list, inertia values list)
    """
    print("\n" + "=" * 50)
    print("ELBOW METHOD ANALYSIS")
    print("=" * 50)
    
    k_values = list(k_range)
    inertias = []
    
    for k in k_values:
        kmeans = run_kmeans(X, k)
        inertias.append(kmeans.inertia_)
        print(f"k = {k:2d}: Inertia (WCSS) = {kmeans.inertia_:,.2f}")
    
    return k_values, inertias


def silhouette_analysis(X: np.ndarray, k_range: range = range(2, 11)) -> Tuple[List[int], List[float]]:
    """
    Perform Silhouette Analysis to find optimal k.
    
    The Silhouette Score measures how similar an object is to its own cluster
    compared to other clusters. Score ranges from -1 to 1:
    - Near +1: Well clustered
    - Near 0: On or very close to the decision boundary
    - Near -1: Might have been assigned to the wrong cluster
    
    Args:
        X: Feature matrix (n_samples, n_features)
        k_range: Range of k values to test
        
    Returns:
        Tuple of (k values list, silhouette scores list)
    """
    print("\n" + "=" * 50)
    print("SILHOUETTE ANALYSIS")
    print("=" * 50)
    
    k_values = list(k_range)
    silhouette_scores = []
    
    for k in k_values:
        kmeans = run_kmeans(X, k)
        score = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(score)
        print(f"k = {k:2d}: Silhouette Score = {score:.4f}")
    
    # Find best k (highest silhouette score)
    best_idx = np.argmax(silhouette_scores)
    best_k = k_values[best_idx]
    best_score = silhouette_scores[best_idx]
    
    print(f"\nBest k by Silhouette: {best_k} (score = {best_score:.4f})")
    
    return k_values, silhouette_scores


def find_optimal_k(k_values: List[int], inertias: List[float], 
                   silhouette_scores: List[float]) -> Dict[str, int]:
    """
    Determine optimal k from Elbow and Silhouette analysis.
    
    For Elbow: Uses the "elbow point" detection based on rate of change.
    For Silhouette: Uses the maximum score.
    
    Args:
        k_values: List of k values tested
        inertias: Corresponding inertia values
        silhouette_scores: Corresponding silhouette scores
        
    Returns:
        Dictionary with optimal k values from each method
    """
    print("\n" + "=" * 50)
    print("OPTIMAL K DETERMINATION")
    print("=" * 50)
    
    results = {}
    
    # Elbow method: find point of maximum curvature
    # Using the "knee" detection - where second derivative changes most
    inertias = np.array(inertias)
    
    # Calculate first and second derivatives
    first_deriv = np.diff(inertias)
    second_deriv = np.diff(first_deriv)
    
    # Find elbow point (where second derivative is maximum)
    elbow_idx = np.argmax(second_deriv) + 1  # +1 because diff reduces length
    elbow_k = k_values[elbow_idx]
    results['elbow'] = elbow_k
    
    # Silhouette: maximum score
    silhouette_k = k_values[np.argmax(silhouette_scores)]
    results['silhouette'] = silhouette_k
    
    print(f"Optimal k by Elbow Method: {elbow_k}")
    print(f"Optimal k by Silhouette Analysis: {silhouette_k}")
    
    return results


def fit_final_model(X: np.ndarray, k: int) -> Tuple[KMeans, np.ndarray]:
    """
    Fit the final k-Means model with the chosen k.
    
    Args:
        X: Feature matrix
        k: Number of clusters
        
    Returns:
        Tuple of (fitted model, cluster labels)
    """
    print(f"\n" + "=" * 50)
    print(f"FITTING FINAL MODEL WITH k = {k}")
    print("=" * 50)
    
    kmeans = run_kmeans(X, k)
    labels = kmeans.labels_
    
    # Print cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    print("\nCluster Distribution:")
    for cluster, count in zip(unique, counts):
        percentage = count / len(labels) * 100
        print(f"  Cluster {cluster}: {count:,} customers ({percentage:.1f}%)")
    
    # Final silhouette score
    final_silhouette = silhouette_score(X, labels)
    print(f"\nFinal Silhouette Score: {final_silhouette:.4f}")
    print(f"Final Inertia: {kmeans.inertia_:,.2f}")
    
    return kmeans, labels


def analyze_clusters(rfm_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """
    Analyze characteristics of each cluster.
    
    Args:
        rfm_df: Original RFM DataFrame (before standardization)
        labels: Cluster labels for each customer
        
    Returns:
        DataFrame with cluster statistics
    """
    print("\n" + "=" * 50)
    print("CLUSTER CHARACTERISTICS ANALYSIS")
    print("=" * 50)
    
    rfm_with_clusters = rfm_df.copy()
    rfm_with_clusters['Cluster'] = labels
    
    # Calculate statistics for each cluster
    cluster_stats = rfm_with_clusters.groupby('Cluster').agg({
        'Recency': ['mean', 'median', 'std'],
        'Frequency': ['mean', 'median', 'std'],
        'Monetary': ['mean', 'median', 'std'],
        'CustomerID': 'count'
    }).round(2)
    
    # Flatten column names
    cluster_stats.columns = ['_'.join(col).strip() for col in cluster_stats.columns.values]
    cluster_stats = cluster_stats.rename(columns={'CustomerID_count': 'Customer_Count'})
    
    print("\nCluster Statistics (Mean values):")
    
    # Create summary for interpretation
    summary = rfm_with_clusters.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    
    # Overall means for comparison
    overall_means = rfm_df[['Recency', 'Frequency', 'Monetary']].mean()
    
    cluster_interpretations = []
    
    for cluster in summary.index:
        row = summary.loc[cluster]
        count = rfm_with_clusters[rfm_with_clusters['Cluster'] == cluster].shape[0]
        
        # Determine characteristics
        recency_level = "Recent" if row['Recency'] < overall_means['Recency'] else "Old"
        frequency_level = "Frequent" if row['Frequency'] > overall_means['Frequency'] else "Rare"
        monetary_level = "High-value" if row['Monetary'] > overall_means['Monetary'] else "Low-value"
        
        interpretation = f"{recency_level}, {frequency_level}, {monetary_level}"
        cluster_interpretations.append({
            'Cluster': cluster,
            'Count': count,
            'Recency_Mean': round(row['Recency'], 1),
            'Frequency_Mean': round(row['Frequency'], 1),
            'Monetary_Mean': round(row['Monetary'], 2),
            'Interpretation': interpretation
        })
        
        print(f"\nCluster {cluster} ({count:,} customers):")
        print(f"  Recency: {row['Recency']:.1f} days ({recency_level})")
        print(f"  Frequency: {row['Frequency']:.1f} orders ({frequency_level})")
        print(f"  Monetary: ${row['Monetary']:,.2f} ({monetary_level})")
        print(f"  → {interpretation}")
    
    interpretation_df = pd.DataFrame(cluster_interpretations)
    
    return interpretation_df


if __name__ == "__main__":
    # Test the module
    from dataset import get_rfm_data
    from preprocessing import preprocess_rfm
    
    rfm_df = get_rfm_data()
    rfm_processed, scaler, info = preprocess_rfm(rfm_df)
    
    X = rfm_processed[['Recency', 'Frequency', 'Monetary']].values
    
    # Run analyses
    k_values, inertias = elbow_analysis(X)
    k_values, silhouette_scores = silhouette_analysis(X)
    
    optimal_k = find_optimal_k(k_values, inertias, silhouette_scores)
    
    # Fit final model
    kmeans, labels = fit_final_model(X, optimal_k['silhouette'])
    
    # Analyze clusters using original RFM values
    interpretation = analyze_clusters(rfm_df, labels)
    print("\n" + "=" * 50)
    print("INTERPRETATION TABLE")
    print("=" * 50)
    print(interpretation.to_string(index=False))
