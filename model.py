# model.py

## This Project is done by Group 11 - Yaashika Murpani, Mahin Narvekar, Krrish Nihalani

# Importing Libraries
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.cluster import KMeans
#from sklearn.externals import joblib


def load_data(file_path):
    """Load the Iris dataset."""
    iris = pd.read_csv("C:/Users/narve/OneDrive/Pictures/Documents/VESP/iris.csv")  # Load Data
    iris.drop('Id', inplace=True, axis=1)  # Drop Id column
    return iris

def visualize_data_distribution(iris):
    """Visualize the data distribution."""
    fig = px.pie(iris, 'Species', color_discrete_sequence=['#491D8B', '#7D3AC1', '#EB548C'], title='Data Distribution', template='plotly')
    fig.show()

def visualize_sepal_length(iris):
    """Visualize Sepal Length distribution."""
    fig = px.box(data_frame=iris, x='Species', y='SepalLengthCm', color='Species', color_discrete_sequence=['#29066B', '#7D3AC1', '#EB548C'], orientation='v')
    fig.show()
    fig = px.histogram(data_frame=iris, x='SepalLengthCm', color='Species', color_discrete_sequence=['#491D8B', '#7D3AC1', '#EB548C'], nbins=50)
    fig.show()

def visualize_sepal_width(iris):
    """Visualize Sepal Width distribution."""
    fig = px.box(data_frame=iris, x='Species', y='SepalWidthCm', color='Species', color_discrete_sequence=['#29066B', '#7D3AC1', '#EB548C'], orientation='v')
    fig.show()
    fig = px.histogram(data_frame=iris, x='SepalWidthCm', color='Species', color_discrete_sequence=['#491D8B', '#7D3AC1', '#EB548C'], nbins=30)
    fig.show()

def visualize_petal_length(iris):
    """Visualize Petal Length distribution."""
    fig = px.box(data_frame=iris, x='Species', y='PetalLengthCm', color='Species', color_discrete_sequence=['#29066B', '#7D3AC1', '#EB548C'], orientation='v')
    fig.show()
    fig = px.histogram(data_frame=iris, x='PetalLengthCm', color='Species', color_discrete_sequence=['#491D8B', '#7D3AC1', '#EB548C'], nbins=30)
    fig.show()

def visualize_petal_width(iris):
    """Visualize Petal Width distribution."""
    fig = px.box(data_frame=iris, x='Species', y='PetalWidthCm', color='Species', color_discrete_sequence=['#29066B', '#7D3AC1', '#EB548C'], orientation='v')
    fig.show()
    fig = px.histogram(data_frame=iris, x='PetalWidthCm', color='Species', color_discrete_sequence=['#491D8B', '#7D3AC1', '#EB548C'], nbins=30)
    fig.show()

def visualize_scatter_plots(iris):
    """Visualize scatter plots for Sepal and Petal dimensions."""
    fig = px.scatter(data_frame=iris, x='SepalLengthCm', y='SepalWidthCm', color='Species', size='PetalLengthCm', template='seaborn', color_discrete_sequence=['#491D8B', '#7D3AC1', '#EB548C'])
    fig.update_layout(width=800, height=600, xaxis=dict(color="#BF40BF"), yaxis=dict(color="#BF40BF"))
    fig.show()

    fig = px.scatter(data_frame=iris, x='PetalLengthCm', y='PetalWidthCm', color='Species', size='SepalLengthCm', template='seaborn', color_discrete_sequence=['#491D8B', '#7D3AC1', '#EB548C'])
    fig.update_layout(width=800, height=600, xaxis=dict(color="#BF40BF"), yaxis=dict(color="#BF40BF"))
    fig.show()

def elbow_method(X):
    """Determine the optimal number of clusters for KMeans using the elbow method."""
    sse = []
    for i in range(1, 9):
        kmeans = KMeans(n_clusters=i, max_iter=300)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)

        # Save the model for the optimal cluster (e.g., if you decide to use 3 clusters)
        if i == 3:  # Change this if you select a different number
            joblib.dump(kmeans, 'kmeans_model.pkl')
    
    fig = px.line(y=sse, template="seaborn", title='Elbow Method')
    fig.update_layout(width=800, height=600, title_font_color="#BF40BF",
                      xaxis=dict(color="#BF40BF", title="Clusters"),
                      yaxis=dict(color="#BF40BF", title="SSE"))
    fig.show()

def main():
    """Main function to execute the program."""
    file_path = "C:/Users/narve/OneDrive/Pictures/Documents/VESP/iris.csv"
    iris = load_data(file_path)
    
    # Set training data
    X = iris.iloc[:, :-1].values  # Features for clustering

    # Visualizations
    visualize_data_distribution(iris)
    visualize_sepal_length(iris)
    visualize_sepal_width(iris)
    visualize_petal_length(iris)
    visualize_petal_width(iris)
    visualize_scatter_plots(iris)

    # K-Means Elbow Method
    elbow_method(X)

if __name__ == "__main__":
    main()
