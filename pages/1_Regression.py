import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt

## Evans Eli Kumah - 10211100312
st.title("📈 Simple Linear Regression")

uploaded_file = st.file_uploader("Upload your regression dataset (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", df.head())

    st.write("### Preprocessing Options")
    if st.checkbox("Drop missing values"):
        df = df.dropna()

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    target_col = st.selectbox("Select target column", options=numeric_cols)

    feature_cols = [col for col in numeric_cols if col != target_col]
    feature_col = st.selectbox("Select ONE feature (X) column", options=feature_cols)

    if target_col and feature_col:
        X = df[[feature_col]]
        y = df[target_col]

        if st.checkbox("Standardize"):
            X = (X - X.mean()) / X.std()

        
        outlier_method = st.radio(
            "Select outlier removal method:",
            ("None", "±3σ", "Percentiles: 5th and 95th")
        )

        if outlier_method == "±3σ":
            # Calculate z-scores for both features and target variable
            z_scores_X = (X - X.mean()) / X.std()
            z_scores_y = (y - y.mean()) / y.std()

            # Combine z-scores for both feature and target, and remove outliers
            mask_X = (z_scores_X.abs() <= 3).all(axis=1)
            mask_y = (z_scores_y.abs() <= 3)

            # Apply the mask for both X and y to remove outliers
            mask = mask_X & mask_y
            X = X[mask]
            y = y[mask]

        elif outlier_method == "Percentiles: 5th and 95th":
            lower_bound = y.quantile(0.05)  # 5th percentile
            upper_bound = y.quantile(0.95)  # 95th percentile

            mask = (y >= lower_bound) & (y <= upper_bound)
            X = X[mask]
            y = y[mask]


        model = LinearRegression()
        model.fit(X, y)
        preds = model.predict(X)

        st.write("### Model Performance")
        st.write(f"Mean Absolute Error (MAE): {mean_absolute_error(y, preds):.2f}")
        st.write(f"R² Score: {r2_score(y, preds):.2f}")
        st.write(f"Mean Squared Error (MSE): {mean_squared_error(y, preds):.2f}")
        st.write(f"Root Mean Squared Error (RMSE): {mean_squared_error(y, preds)**0.5:.2f}")
        st.write(f"Mean Absolute Percentage Error (MAPE): {mean_absolute_percentage_error(y, preds)*100:.2f}%")

        st.write("### Regression Line Visualization")
        fig, ax = plt.subplots()
        ax.scatter(X, y, label='Actual')
        ax.plot(X, preds, color='red', label='Regression Line')
        ax.set_xlabel(feature_col)
        ax.set_ylabel(target_col)
        ax.legend()
        st.pyplot(fig)

        st.write("### Predict with Custom Input")
        user_input = st.number_input(f"Enter value for {feature_col}", value=float(X[feature_col].mean()))
        if st.button("Predict"):
            prediction = model.predict([[user_input]])
            st.success(f"Predicted {target_col}: {prediction[0]:.2f}")
