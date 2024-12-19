import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import mlflow
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_process_data(file_path):
    """
    Load the data and preprocess it for clustering.
    """
    logger.info("Loading data...")
    df = pd.read_csv(file_path)
    df = df.drop(columns=["Country", "ProductName", "TransactionNo", "ProductNo", "Date"])
    df["CustomerNo"] = df["CustomerNo"].fillna(df["CustomerNo"].mode()[0])
    
    # Feature Engineering
    df["Total_Spent"] = df["Price"] * df["Quantity"]
    df["Frequency"] = df["CustomerNo"].map(df["CustomerNo"].value_counts())
    total_spent = df.groupby("CustomerNo")["Total_Spent"].transform("sum")
    df["Total_Spent"] = total_spent
    df["Average_Spent"] = df["Total_Spent"] / df["Frequency"]
    df = df.drop(columns=["Price", "Quantity"])
    df_new = df.drop_duplicates(subset="CustomerNo")[["Total_Spent", "Frequency", "Average_Spent"]]
    
    # Standardize Data
    logger.info("Standardizing data...")
    scaler = StandardScaler()
    df_new[["Total_Spent", "Frequency", "Average_Spent"]] = scaler.fit_transform(
        df_new[["Total_Spent", "Frequency", "Average_Spent"]])
    return df_new

def perform_clustering(df_new, n_clusters):
    """
    Perform KMeans clustering and log the results.
    """
    logger.info("Starting KMeans clustering...")
    with mlflow.start_run():
        # Apply KMeans
        model = KMeans(n_clusters=n_clusters, random_state=2024, init="k-means++")
        df_new["Cluster"] = model.fit_predict(df_new)
        
        # Calculate Silhouette Score
        silhouette = silhouette_score(df_new[["Total_Spent", "Frequency", "Average_Spent"]], df_new["Cluster"])
        logger.info(f"Silhouette Score: {silhouette}")
        
        # Log metrics and model in MLflow
        mlflow.log_param("n_clusters", n_clusters)
        mlflow.log_metric("silhouette_score", silhouette)
        mlflow.sklearn.log_model(model, "kmeans_model")
        
        return df_new, model, silhouette

def visualize_clusters(df_new):
    """
    Visualize the clusters in a 3D scatter plot.
    """
    logger.info("Visualizing clusters...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    scatter = ax.scatter(df_new["Total_Spent"], df_new["Frequency"], df_new["Average_Spent"],
                         c=df_new["Cluster"], cmap="Set1", s=50)
    ax.set_xlabel("Total Spent")
    ax.set_ylabel("Frequency")
    ax.set_zlabel("Average Spent")
    ax.set_title("KMeans Clustering in 3D Space")
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.show()

def main():
    file_path = "Sales Transaction v.4a.csv"  # Replace with your file path
    logger.info("Clustering pipeline started.")
    
    # Load and process data
    df_new = load_and_process_data(file_path)
    
    # Perform clustering
    n_clusters = 5  # You can tune this
    df_new, model, silhouette = perform_clustering(df_new, n_clusters)
    
    # Visualize results
    visualize_clusters(df_new)

    logger.info("Clustering pipeline completed successfully.")

if __name__ == "__main__":
    main()
