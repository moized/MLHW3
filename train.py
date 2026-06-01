"""
Main Training Script for k-Means Customer Segmentation
Performs RFM analysis, preprocessing, clustering, and visualization.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime

# Import project modules
from dataset import get_rfm_data
from preprocessing import preprocess_rfm, calculate_all_skewness
from clustering import (
    elbow_analysis, 
    silhouette_analysis, 
    find_optimal_k,
    fit_final_model, 
    analyze_clusters
)
from utils import (
    plot_rfm_histograms,
    plot_elbow_curve,
    plot_silhouette_curve,
    plot_combined_analysis,
    apply_pca,
    plot_pca_clusters,
    plot_cluster_characteristics,
    ensure_results_dir
)


def save_results_to_file(results: dict, filepath: str) -> None:
    """
    Save analysis results to a text file.
    
    Args:
        results: Dictionary containing all results
        filepath: Path to save the file
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("K-MEANS CUSTOMER SEGMENTATION - ANALYSIS RESULTS\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        
        # Dataset info
        f.write("1. DATASET INFORMATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total customers analyzed: {results['n_customers']:,}\n")
        f.write(f"Dataset: Online Retail II (UCI)\n\n")
        
        # RFM Statistics
        f.write("2. RFM STATISTICS (Original Data)\n")
        f.write("-" * 40 + "\n")
        f.write(results['rfm_stats'].to_string() + "\n\n")
        
        # Skewness Analysis
        f.write("3. SKEWNESS ANALYSIS\n")
        f.write("-" * 40 + "\n")
        f.write("Initial Skewness (before transformation):\n")
        for feature, value in results['initial_skewness'].items():
            f.write(f"  {feature}: {value:.4f}\n")
        
        f.write(f"\nLog transformation applied: {results['log_applied']}\n")
        
        if results['log_applied']:
            f.write("\nSkewness after log transformation:\n")
            for feature, value in results['transformed_skewness'].items():
                f.write(f"  {feature}: {value:.4f}\n")
        f.write("\n")
        
        # Elbow Analysis
        f.write("4. ELBOW METHOD RESULTS\n")
        f.write("-" * 40 + "\n")
        for k, inertia in zip(results['k_values'], results['inertias']):
            f.write(f"  k = {k:2d}: Inertia = {inertia:,.2f}\n")
        f.write(f"\nOptimal k (Elbow): {results['optimal_k_elbow']}\n\n")
        
        # Silhouette Analysis
        f.write("5. SILHOUETTE ANALYSIS RESULTS\n")
        f.write("-" * 40 + "\n")
        for k, score in zip(results['k_values'], results['silhouette_scores']):
            f.write(f"  k = {k:2d}: Silhouette Score = {score:.4f}\n")
        f.write(f"\nOptimal k (Silhouette): {results['optimal_k_silhouette']}\n\n")
        
        # Final Model
        f.write("6. FINAL MODEL\n")
        f.write("-" * 40 + "\n")
        f.write(f"Selected k: {results['final_k']}\n")
        f.write(f"Final Silhouette Score: {results['final_silhouette']:.4f}\n")
        f.write(f"Final Inertia: {results['final_inertia']:,.2f}\n\n")
        
        # Cluster Interpretation
        f.write("7. CLUSTER INTERPRETATION\n")
        f.write("-" * 40 + "\n")
        f.write(results['cluster_interpretation'].to_string(index=False) + "\n\n")
        
        # PCA Info
        f.write("8. PCA ANALYSIS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Components: 2\n")
        for i, ratio in enumerate(results['pca_variance_ratio']):
            f.write(f"  PC{i+1}: {ratio:.4f} ({ratio*100:.2f}%)\n")
        f.write(f"  Total variance explained: {sum(results['pca_variance_ratio'])*100:.2f}%\n\n")
        
        # Execution time
        f.write("9. EXECUTION TIME\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total time: {results['execution_time']:.2f} seconds\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")
    
    print(f"Results saved to: {filepath}")


def main():
    """
    Main function to run the complete k-Means clustering analysis.
    """
    start_time = time.time()
    
    print("=" * 70)
    print("K-MEANS CUSTOMER SEGMENTATION ANALYSIS")
    print("Online Retail II Dataset - RFM Analysis")
    print("=" * 70)
    
    # Create results directory
    results_dir = ensure_results_dir("results")
    
    # Dictionary to store all results
    results = {}
    
    # =========================================================================
    # STEP 1: Load and prepare RFM data
    # =========================================================================
    print("\n" + "#" * 70)
    print("STEP 1: FEATURE ENGINEERING (RFM CALCULATION)")
    print("#" * 70)
    
    rfm_df = get_rfm_data()
    results['n_customers'] = len(rfm_df)
    results['rfm_stats'] = rfm_df[['Recency', 'Frequency', 'Monetary']].describe()
    
    # Save original RFM data
    rfm_df.to_csv(os.path.join(results_dir, "rfm_data.csv"), index=False)
    print(f"\nRFM data saved to: {results_dir}/rfm_data.csv")
    
    # Plot histograms for original RFM
    plot_rfm_histograms(
        rfm_df, 
        "(Before Transformation)", 
        os.path.join(results_dir, "rfm_histograms_original.png")
    )
    
    # =========================================================================
    # STEP 2: Distribution Check and Preprocessing
    # =========================================================================
    print("\n" + "#" * 70)
    print("STEP 2: DISTRIBUTION CHECK AND PREPROCESSING")
    print("#" * 70)
    
    rfm_processed, scaler, preprocess_info = preprocess_rfm(rfm_df)
    
    results['initial_skewness'] = preprocess_info['initial_skewness']
    results['log_applied'] = preprocess_info['log_applied']
    results['transformed_skewness'] = preprocess_info['transformed_skewness']
    
    # Plot histograms after transformation (before standardization)
    if preprocess_info['log_applied']:
        rfm_log = rfm_df.copy()
        rfm_log[['Recency', 'Frequency', 'Monetary']] = np.log1p(
            rfm_df[['Recency', 'Frequency', 'Monetary']]
        )
        plot_rfm_histograms(
            rfm_log,
            "(After Log Transform)",
            os.path.join(results_dir, "rfm_histograms_log.png")
        )
    
    # =========================================================================
    # STEP 3: K-Means Clustering with Elbow and Silhouette Analysis
    # =========================================================================
    print("\n" + "#" * 70)
    print("STEP 3: K-MEANS CLUSTERING ANALYSIS")
    print("#" * 70)
    
    # Prepare feature matrix
    X = rfm_processed[['Recency', 'Frequency', 'Monetary']].values
    
    # Run Elbow Analysis
    k_values, inertias = elbow_analysis(X, k_range=range(2, 11))
    results['k_values'] = k_values
    results['inertias'] = inertias
    
    # Run Silhouette Analysis
    _, silhouette_scores = silhouette_analysis(X, k_range=range(2, 11))
    results['silhouette_scores'] = silhouette_scores
    
    # Find optimal k
    optimal_k = find_optimal_k(k_values, inertias, silhouette_scores)
    results['optimal_k_elbow'] = optimal_k['elbow']
    results['optimal_k_silhouette'] = optimal_k['silhouette']
    
    # Plot Elbow curve
    plot_elbow_curve(
        k_values, inertias, 
        optimal_k=optimal_k['elbow'],
        save_path=os.path.join(results_dir, "elbow_plot.png")
    )
    
    # Plot Silhouette curve
    plot_silhouette_curve(
        k_values, silhouette_scores,
        optimal_k=optimal_k['silhouette'],
        save_path=os.path.join(results_dir, "silhouette_plot.png")
    )
    
    # Plot combined analysis
    plot_combined_analysis(
        k_values, inertias, silhouette_scores,
        optimal_k['elbow'], optimal_k['silhouette'],
        save_path=os.path.join(results_dir, "combined_analysis.png")
    )
    
    # Save analysis data
    analysis_df = pd.DataFrame({
        'k': k_values,
        'Inertia': inertias,
        'Silhouette_Score': silhouette_scores
    })
    analysis_df.to_csv(os.path.join(results_dir, "clustering_analysis.csv"), index=False)
    
    # =========================================================================
    # STEP 4: Fit Final Model and Analyze Clusters
    # =========================================================================
    print("\n" + "#" * 70)
    print("STEP 4: CLUSTER ANALYSIS AND VISUALIZATION")
    print("#" * 70)
    
    # Use silhouette optimal k for final model
    final_k = optimal_k['silhouette']
    results['final_k'] = final_k
    
    # Fit final model
    from sklearn.metrics import silhouette_score
    kmeans, labels = fit_final_model(X, final_k)
    results['final_silhouette'] = silhouette_score(X, labels)
    results['final_inertia'] = kmeans.inertia_
    
    # Analyze clusters (using original RFM values for interpretability)
    interpretation_df = analyze_clusters(rfm_df, labels)
    results['cluster_interpretation'] = interpretation_df
    
    # Save cluster assignments
    rfm_with_clusters = rfm_df.copy()
    rfm_with_clusters['Cluster'] = labels
    rfm_with_clusters.to_csv(os.path.join(results_dir, "customer_segments.csv"), index=False)
    print(f"\nCustomer segments saved to: {results_dir}/customer_segments.csv")
    
    # Save interpretation
    interpretation_df.to_csv(os.path.join(results_dir, "cluster_interpretation.csv"), index=False)
    
    # Plot cluster characteristics
    plot_cluster_characteristics(
        interpretation_df,
        save_path=os.path.join(results_dir, "cluster_characteristics.png")
    )
    
    # =========================================================================
    # STEP 5: PCA Visualization
    # =========================================================================
    print("\n" + "#" * 70)
    print("STEP 5: PCA DIMENSIONALITY REDUCTION AND VISUALIZATION")
    print("#" * 70)
    
    # Apply PCA
    X_pca, pca_model = apply_pca(X, n_components=2)
    results['pca_variance_ratio'] = pca_model.explained_variance_ratio_.tolist()
    
    # Transform cluster centers to PCA space
    cluster_centers_pca = pca_model.transform(kmeans.cluster_centers_)
    
    # Plot PCA clusters
    plot_pca_clusters(
        X_pca, labels,
        cluster_centers_pca=cluster_centers_pca,
        save_path=os.path.join(results_dir, "pca_clusters.png")
    )
    
    # =========================================================================
    # Save Results
    # =========================================================================
    execution_time = time.time() - start_time
    results['execution_time'] = execution_time
    
    save_results_to_file(results, os.path.join(results_dir, "metrics.txt"))
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\nExecution time: {execution_time:.2f} seconds")
    print(f"\nGenerated files in '{results_dir}/':")
    for file in os.listdir(results_dir):
        print(f"  - {file}")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total customers: {results['n_customers']:,}")
    print(f"Optimal k (Elbow): {results['optimal_k_elbow']}")
    print(f"Optimal k (Silhouette): {results['optimal_k_silhouette']}")
    print(f"Final k used: {results['final_k']}")
    print(f"Final Silhouette Score: {results['final_silhouette']:.4f}")
    print("\nCluster Interpretation:")
    print(interpretation_df[['Cluster', 'Count', 'Interpretation']].to_string(index=False))
    
    return results


if __name__ == "__main__":
    main()
