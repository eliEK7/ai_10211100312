import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px

## Evans Eli Kumah - 10211100312

st.title("🔗 K-Means Clustering")

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", df.head())

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    selected_features = st.multiselect("Select features for clustering (at least 2)", numeric_cols)

    if len(selected_features) >= 2:
        X = df[selected_features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        st.sidebar.write("### Clustering Controls")
        n_clusters = st.sidebar.slider("Select number of clusters", min_value=2, max_value=10, value=3)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        centroids = kmeans.cluster_centers_

        df['Cluster'] = cluster_labels

        # --- Cluster Summary ---
        st.write("### 📊 Cluster Summary")

        cluster_counts = df['Cluster'].value_counts().sort_index()
        st.write("**Cluster Sizes:**")
        st.write(cluster_counts.rename("Number of Points"))

        st.write("**Feature Means per Cluster:**")
        cluster_summary = df.groupby('Cluster')[selected_features].mean()
        st.dataframe(cluster_summary)

        # Downloadable summary
        csv_summary = cluster_summary.reset_index().to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Cluster Summary",
            data=csv_summary,
            file_name='cluster_summary.csv',
            mime='text/csv'
        )


        st.write("### Clustered Dataset")
        st.dataframe(df)

        st.download_button("Download Clustered Data", df.to_csv(index=False), file_name="clustered_data.csv")

        st.write("### Cluster Visualization")
        if len(selected_features) == 2:
            fig, ax = plt.subplots()
            scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster_labels, cmap='viridis')
            ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X', label='Centroids')
            ax.set_xlabel(selected_features[0])
            ax.set_ylabel(selected_features[1])
            ax.legend()
            st.pyplot(fig)

        elif len(selected_features) >= 3:
            fig = px.scatter_3d(
                df,
                x=selected_features[0],
                y=selected_features[1],
                z=selected_features[2],
                color=df['Cluster'].astype(str),
                title="3D Cluster Visualization",
            )
            st.plotly_chart(fig)
        else:
            st.info("Select at least two numeric features for visualization.")
