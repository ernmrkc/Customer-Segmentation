from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tabulate import tabulate
from typing import Tuple, List, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

class ClusteringEngine:
    def __init__(self):
        """
        Initializes the DataPreprocessor class.
        """
        pass    
        
    def findOptimumClusterNumber(self, dataframe: pd.DataFrame, max_clusters: int) -> None:
        """
        Finds and plots the optimal number of clusters based on silhouette scores and the elbow method.

        Parameters:
        - dataframe (pd.DataFrame): The DataFrame containing the features for clustering.
        - max_clusters (int): The maximum number of clusters to consider for finding the optimal number.

        Note:
        This function utilizes the 'clusterSilhouetteAndElbowMethod' function to compute and plot clustering scores.
        """
        def clusterSilhouetteAndElbowMethod(x: pd.DataFrame, max_clusters: int) -> Tuple[List[float], List[int], List[float]]:
            """
            Calculates silhouette scores and sum of squared distances for a range of cluster numbers to aid in determining the optimal number of clusters.

            Parameters:
            - x (pd.DataFrame): The DataFrame containing the features for clustering.
            - max_clusters (int): The maximum number of clusters to test.

            Returns:
            - Tuple containing three lists: silhouette scores, cluster numbers, and sum of squared distances.
            """
            silhouette_scores = []
            cluster_numbers = []
            sum_of_squared_distances = []

            for NoC in range(2, max_clusters+1):  # Starting from 2 since silhouette score cannot be calculated for a single cluster
                model = KMeans(n_clusters=NoC, random_state=42)
                model.fit(x)
                labels = model.predict(x)
                sum_of_squared_distances.append(model.inertia_)
                silhouette = silhouette_score(x, labels)
                print(f"Number of clusters: {NoC}, silhouette: {silhouette:.2f}")
                silhouette_scores.append(silhouette)
                cluster_numbers.append(NoC)

            # Plotting the elbow plot
            plt.figure(figsize=(8, 6))
            plt.plot(cluster_numbers, sum_of_squared_distances, marker="x")
            plt.xlabel("Number of Clusters")
            plt.ylabel("Sum of Squared Distances")
            plt.title("Elbow Method Showing the Optimal k")
            plt.grid(True)
            plt.show()

            return silhouette_scores, cluster_numbers, sum_of_squared_distances
        
        # Assuming clusterSilhouetteAndElbowMethod is defined as shown earlier
        silhouette_scores, cluster_numbers, sum_of_squared_distances = clusterSilhouetteAndElbowMethod(dataframe, max_clusters)

        # You might want to print or further analyze silhouette_scores, cluster_numbers, and sum_of_squared_distances
        # For example, print the highest silhouette score and its corresponding cluster number:
        max_silhouette = max(silhouette_scores)
        optimal_clusters = cluster_numbers[silhouette_scores.index(max_silhouette)]
        print(f"Optimal number of clusters based on silhouette score: {optimal_clusters} with a silhouette score of: {max_silhouette:.2f}")

        # Note: The plots are generated within the 'clusterSilhouetteAndElbowMethod' function.
        
    def fitAndPredictAll(self, myAlgorithms: List[Any], X: np.ndarray) -> None:
        """
        aff = AffinityPropagation(damping=0.9)
        agg = AgglomerativeClustering(n_clusters=3)
        br = Birch(threshold= 0.01, n_clusters= 4)
        db = DBSCAN(eps= 0.30, min_samples=9)
        mbkm = MiniBatchKMeans(n_clusters=3)
        ms = MeanShift()
        opt = OPTICS(eps= 0.8, min_samples=10)
        spec = SpectralClustering(n_clusters= 4)
        gmix = GaussianMixture(n_components=3)
        km = KMeans(n_clusters=4)

        # Add the algorithms you define to the array
        myAlgorithmArray = [aff, agg, br, db, mbkm, ms, opt, spec, gmix, km]
        
        Trains given clustering algorithms with the dataset and calculates silhouette scores for the clustering results.

        Parameters:
        - myAlgorithms (List[Any]): List of instantiated clustering algorithms.
        - X (np.ndarray): The dataset for clustering (features).

        Note: This function prints the silhouette score for each algorithm in a tabular format.
        """
        predictions = []  # Store algorithm names and their silhouette scores

        for algo in myAlgorithms:
            algo_name = type(algo).__name__  # Get the algorithm name
            try:
                labels = algo.fit_predict(X)  # Fit the model and predict clusters
                silhouette = silhouette_score(X, labels)  # Calculate silhouette score
                predictions.append([algo_name, silhouette])
            except Exception as e:
                print(f"An error occurred during {algo_name} process: {e}")

        # Print the results in a tabular format
        print(tabulate(predictions, headers=["Algorithm", "Silhouette Score"]))
