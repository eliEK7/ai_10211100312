import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import pandas as pd

## Evans Eli Kumah - 10211100312

st.title("Neural Network Classifier")

# Upload dataset
uploaded_file = st.file_uploader("Upload classification dataset (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", df.head())

    st.write("### Preprocessing")
    target_col = st.selectbox("Select target column (Target must be Categorical)", options=df.columns)

    if target_col:
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Check if the target column is categorical
        if y.dtype != 'object' and not pd.api.types.is_categorical_dtype(y):
            st.error("⚠️ The target column must be categorical for classification tasks. Please select a categorical target column.")
        else:
            # Encode categorical target labels if necessary
            if y.dtype == 'object':
                encoder = LabelEncoder()
                y = encoder.fit_transform(y)

            # Keep only numeric features for now
            X = X.select_dtypes(include=np.number)

            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Hyperparameters
            st.sidebar.header("Hyperparameters")
            epochs = st.sidebar.slider("Epochs", 1, 100, 20)
            learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.1, 0.001, format="%.4f")

            # Train-test split
            X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            # Build model
            model = Sequential([
                Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
                Dense(32, activation='relu'),
                Dense(len(np.unique(y)), activation='softmax')
            ])
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            # Train model
            with st.spinner("Training model..."):
                history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, verbose=0)

            # Plot loss and accuracy
            st.write("### 📉 Training Progress")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            ax1.plot(history.history['loss'], label='Train Loss')
            ax1.plot(history.history['val_loss'], label='Val Loss')
            ax1.set_title('Loss')
            ax1.legend()

            ax2.plot(history.history['accuracy'], label='Train Accuracy')
            ax2.plot(history.history['val_accuracy'], label='Val Accuracy')
            ax2.set_title('Accuracy')
            ax2.legend()

            st.pyplot(fig)

            # Additional metrics: confusion matrix and classification report
            y_pred = model.predict(X_val)
            y_pred_classes = np.argmax(y_pred, axis=1)

            # Confusion Matrix
            cm = confusion_matrix(y_val, y_pred_classes)
            fig_cm, ax_cm = plt.subplots(figsize=(6, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
            ax_cm.set_xlabel('Predicted')
            ax_cm.set_ylabel('True')
            ax_cm.set_title('Confusion Matrix')
            st.pyplot(fig_cm)

            # Generate the classification report
            class_report = classification_report(y_val, y_pred_classes, target_names=encoder.classes_, labels=np.unique(y), zero_division=0)

            # Convert the classification report to a DataFrame for better alignment
            class_report_dict = classification_report(y_val, y_pred_classes, target_names=encoder.classes_, labels=np.unique(y), zero_division=0, output_dict=True)

            # Convert the report into a pandas DataFrame
            class_report_df = pd.DataFrame(class_report_dict).transpose()

            # Display the classification report as a pandas DataFrame
            st.write("### 📊 Classification Report")
            st.write(class_report_df)


            # Upload test sample for prediction
            st.write("### 🧪 Predict on Custom Sample")
            test_file = st.file_uploader("Upload new sample(s) for prediction", type=["csv"], key="predict_file")
            if test_file:
                test_df = pd.read_csv(test_file)
                st.write("### Test Dataset Preview", test_df.head())

                # Ensure consistent column names (strip leading/trailing spaces and make lowercase for comparison)
                test_df.columns = test_df.columns.str.strip()  # Strip spaces
                X_columns_stripped = X.columns.str.strip()  # Strip spaces from training data columns


                # Check if the test data contains the same features as the training data (ignoring order)
                if set(test_df.columns) != set(X_columns_stripped):
                    st.error("❌ The uploaded sample has different features than the training data. Please upload a sample with the same features.")
                else:
                    # Drop the target column in the test dataset if present (to prevent any mismatch)
                    test_X = test_df[X_columns_stripped]  # Only keep the feature columns that match training data
                    test_X_scaled = scaler.transform(test_X)  # Ensure the test data is scaled correctly

                    # Make predictions
                    predictions = model.predict(test_X_scaled)
                    predicted_classes = np.argmax(predictions, axis=1)

                    # If labels were encoded, inverse the transformation
                    if 'encoder' in locals():
                        predicted_classes = encoder.inverse_transform(predicted_classes)

                    # Display predictions
                    result_df = test_df.copy()  # Create a new DataFrame to store predictions
                    result_df["Prediction"] = predicted_classes  # Add the prediction column
                    st.write("### Predictions", result_df)

                    # Download predictions
                    st.download_button(
                        "📥 Download Predictions",
                        data=result_df.to_csv(index=False).encode("utf-8"),
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
