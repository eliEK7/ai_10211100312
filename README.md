# AI & ML Project

### Table of Contents
- [Overview](#overview)
  - [Features](#features)
  - [Tools & Libraries](#tools--libraries)
- [Task Descriptions](#task-descriptions)
  - [Regression Problem](#regression-problem)
  - [Clustering Task](#clustering-task)
  - [Neural Network Task](#neural-network-task)
  - [Large Language Model (LLM) Q&A](#large-language-model-llm-qa)
- [How to Use](#how-to-use)
  - [Regression](#regression)
  - [Clustering](#clustering)
  - [Neural Network](#neural-network)
  - [Large Language Model (LLM)](#large-language-model-llm)
- [Documentation](#documentation)

## Author Information
    Evans Eli Kumah - 10211100312
    evans.kumah@acity.edu.gh

## Overview
This project involves the development of a unified dashboard using Streamlit to handle multiple machine learning tasks, including regression, clustering, neural networks, and large language model (LLM)-based question and answer functionality. The dashboard allows users to upload datasets, train models, visualize results, and make predictions.

**Features**
- Regression Task: Predict continuous variables based on input data (e.g., house prices).
- Clustering Task: Perform clustering using K-Means and visualize results.
- Neural Network Task: Train and use a neural network for classification tasks.
- LLM Task: Use an open-source pre-trained LLM to perform Q&A from custom data (such as budget documents or election results).

#### Tools & Libraries
- Streamlit: Used for the interactive web app.
- Pandas: Data processing and manipulation.
- Scikit-learn: For machine learning models such as regression and clustering.
- TensorFlow/PyTorch: For building neural networks.
- Mistral-7B-Instruct: Open-source LLM for NLP tasks.
- PyMuPDF: For extracting text from PDF documents.

## Task Descriptions
**Regression Problem**
- Functionality:
    - Dataset Upload: Users can upload a CSV file with regression data. The user will specify the target column.
    - Linear Regression Model: The model will predict a continuous variable, such as house prices, based on user-provided features.

- Results:
    - Display performance metrics (e.g., Mean Absolute Error, R² score).
    - Scatter plot of predictions vs. actual values.
    - Custom Predictions: Users can input custom data to make predictions.

- Features:
    - Dataset Preview: Display the first few rows of the dataset.
    - Data Preprocessing Options: Allow users to perform basic data preprocessing (e.g., handling missing values).
    - Regression Line Visualization: Show the regression line on a plot to illustrate model fitting.

**Clustering Task**
- Functionality:
    - Dataset Upload: Users can upload a dataset with multiple features.
    - K-Means Clustering: The model will group data points into clusters based on their features.
    - Cluster Visualization: Display interactive 2D or 3D scatter plots of clustered data.

- Results:
    - Display cluster assignments and centroids.
    - Interactive visualization of clustered data.
    - Downloadable clustered dataset with assignments.

- Features:
    - Dataset Preview: Show the first few rows of the uploaded data.
    - Interactive Cluster Selection: Allow users to adjust the number of clusters (k).
    - Cluster Statistics: Display metrics like inertia and silhouette score.

**Neural Network Task**
- Functionality:
    - Dataset Upload: Users can upload a dataset for classification tasks with specified target column.
    - Neural Network Model: Build and train a Feedforward Neural Network using TensorFlow/PyTorch.
    - Model Training: Train the model with user-defined parameters.

- Results:
    - Display training and validation metrics (accuracy, loss).
    - Show real-time training progress.
    - Enable custom predictions on new data.

- Features:
    - Training Visualization: Real-time plots of accuracy and loss metrics.
    - Hyperparameter Configuration: Adjust learning rate, epochs, and architecture.
    - Model Performance Analysis: Show confusion matrix and classification report.

**Large Language Model (LLM) Q&A**
- Functionality:
    - Document Input: Use pre-trained Mistral-7B-Instruct model with 2025 Ghana Budget Statement.
    - Question Processing: Process user questions using RAG approach.
    - Answer Generation: Generate contextual answers from the document.

- Results:
    - Display generated answers in real-time.
    - Show confidence scores for answers.
    - Highlight relevant document sections.

- Features:
    - Text Input Interface: Clean input box for user questions.
    - Answer Quality Metrics: Display confidence scores and relevance indicators.
    - Context Window: Show the document context used for answering.

## How to Use
 A comprehensive guide to using the Streamlit application, covering all aspects of the project. It includes clear instructions on how to use each feature of the app.
 
 Note: **As part of the project files are CSV samples in the test directory that can be used to evaluate the functionality of the application. Each CSV is formatted for a particular app**

### Regression
This is a simple linear regression web app built using Streamlit. It allows you to upload a CSV dataset, select a numeric feature and a target variable, preprocess the data, train a linear regression model, visualize the results, and make predictions based on custom input.

1. Upload your dataset
    - Click on "Browse files" to upload a CSV file.
    - Ensure your dataset includes at least two numeric columns.
    - The target column should represent the continuous variable you want to predict.
    - The feature column should contain the input values used for prediction.

2. Preview your data
    - The first five rows of your uploaded dataset will be displayed for quick review.

3. Handle missing values
    - Tick the "Drop missing values" checkbox to automatically remove rows with missing data.
    - Use this option if your dataset contains blanks that might affect model accuracy.

4. Select variables
    - Use the dropdown to choose your target column (what the model should predict).
    - Choose a single feature column (what the model should learn from).
    - Only numeric columns will be available.

5. Standardize your data (optional)
    - Tick the "Standardize" checkbox to scale the selected feature.
    - This sets the mean to 0 and standard deviation to 1.

6. Remove outliers (optional)
    - Choose one of the two methods using the radio buttons:
    - "±3σ" removes data points with z-scores beyond ±3.
    - "Percentiles: 5th and 95th" keeps only data within the central 90% of the distribution.
    - Only one method can be applied at a time.
    - Choose "None" to skip outlier removal.
    - Note: Outlier removal applies to both the feature and target variable.

7. Train the model
    - The app trains a simple linear regression model using your selected feature and target.
    - Model performance is shown using these metrics:
    - Mean Absolute Error (MAE)
    - R² Score
    - Mean Squared Error (MSE)
    - Root Mean Squared Error (RMSE)
    - Mean Absolute Percentage Error (MAPE)

8. View the regression line
    - A scatter plot displays the actual vs. predicted values.
    - The red line represents the fitted regression model.

9. Make a prediction
    - Use the input box to enter a custom value for the selected feature.
    - Click the "Predict" button to get the predicted value for the target.
    - The result will display below the button.

    **Warnings:**
    - The app supports only one feature column at a time.
    - Upload only CSV files with numeric data for this version.
    - Avoid using datasets with non-numeric values in the selected columns.
    - Data points removed due to outlier filtering may affect model performance.


### Clustering
This app uses the K-Means Clustering algorithm to group data points based on similarities across numeric features. It helps you explore patterns in your dataset by dividing it into distinct clusters. You can adjust the number of clusters, view summaries, and download results for further analysis.

1. Upload your dataset
    - Upload a CSV file containing your dataset.
    - Your file must include at least two numeric columns.
    - Avoid uploading non-numeric or text-heavy files.

2. Select features
    - Select the numeric features you want to use for clustering.
    - You must select at least two columns.
    - Selecting three or more enables 3D visualizations.

3. Configure clusters
    - Use the slider in the sidebar to choose the number of clusters.
    - The default is 3, but you can set it between 2 and 10.
    - K-Means will group the data points based on the selected features.

4. View summary
    - You will see how many points belong to each cluster.
    - A table shows the average values of each feature in every cluster.

5. Review results
    - The original dataset now includes a new column labeled "Cluster."
    - Each row is assigned to a group based on similarity.

6. Explore visualizations
    - If you selected exactly two features, a 2D scatter plot appears.
    - Cluster centroids are marked with red Xs.
    - If you selected three or more features, a 3D scatter plot is shown.
    - Each point is colored based on its cluster.

7. Export data
    - You can download the cluster summary as a CSV file.
    - You can also export the full dataset with cluster labels.

    **Warnings:**
    - Do not select fewer than two features.
    - Standardize your features beforehand if the scale varies significantly.
    - Clustering results may differ depending on feature selection and cluster count.
    - Upload only clean, properly formatted CSV files.


### Neural Network
This app lets you train and use a simple neural network for classification tasks. You upload your dataset, pick the target column, adjust hyperparameters, view training progress, and predict outcomes for new data.

1. Upload your dataset
    - Upload a CSV file with your classification dataset.
    - Include numeric feature columns.
    - Use a categorical column as the target (e.g., species).

2. Select target
    - Choose the target column from the dropdown.
    - The target must contain categories, not numbers.
    - If the target is not categorical, the app will stop and alert you.

3. Data preprocessing
    - The app encodes the target labels automatically.
    - It keeps only numeric feature columns.
    - It standardizes the data for better model performance.

4. Configure model
    - Use the sidebar to adjust hyperparameters.
    - Set the number of training epochs.
    - Choose a learning rate.

5. Model architecture
    - The model is a feedforward neural network with three layers.
    - It uses ReLU and Softmax activations.
    - It trains on 80% of your data and validates on 20%.

6. Training progress
    - The training process runs silently in the background.
    - You will see two graphs: one for accuracy, one for loss.
    - Both graphs show training and validation performance.

7. View results
    - The app displays a confusion matrix.
    - You can quickly compare predicted vs. actual classes.
    - A classification report shows precision, recall, and F1-score.

8. Make predictions
    - Upload a new CSV file to test the trained model.
    - The file must contain the same feature columns as the training data.
    - Column names must match exactly.

9. Review predictions
    - The app scales and processes your new data automatically.
    - It uses the trained model to make predictions.
    - The results appear in a table with a new "Prediction" column.

10. Export results
    - You can download the predictions as a CSV file.
    - Use this for further analysis or integration.

    **Warnings:**
    - Always select a categorical column as the target.
    - Ensure feature names in your test file match the training data.
    - Do not include the target column in your test file.
    - The model does not save across sessions.
    - Re-upload your data and retrain if you refresh the page.


### Large Language Model (LLM)
The Large Language Model (LLM) enables users to interact with the 2025 Budget Statement and Economic Policy of the Government of Ghana through natural language queries. By leveraging a pre-trained model, such as Mistral 7B-Instruct, the system processes your questions and provides real-time, context-aware answers based on the content of the document.

1. Enter Your Question
You will see a text input box labeled "Ask a question:".
Type your question in the input box. For example, you can ask:
"What are the main economic policies for 2025?"
"How is the government allocating funds for healthcare in the 2025 budget?"

3. Submit Your Question
After typing your question, press Enter or click the Submit button (if available).

4. View the Response
The system will process your question in real-time and display the answer from the model on the screen.

The model's response will appear below your question, clearly labeled as the Budget Bot's reply.

5. See Chat History
As you continue asking questions, the conversation history will be displayed in a chat format.

Your queries will appear on the right side, and the model's responses will appear on the left side.

6. Repeat the Process
You can continue asking questions about the budget or any other topic from the document.

The system will provide answers based on the context from the 2025 Ghana Budget Statement.

7. Real-Time Responses
The responses are generated and displayed in real-time as the model processes the input.

This allows you to interact seamlessly without waiting long periods for answers.



### Large Language Model (LLM)
This app enables interaction with the 2025 Ghana Budget Statement through natural language queries. Using the Mistral-7B-Instruct model, it processes questions and provides context-aware answers from the document content in real-time.

1. Enter question
    - Type your question in the labeled input box.
    - Questions can be about any topic covered in the budget.
    - Use natural language for your queries.

2. Submit Your Question
    - After typing your question, press Enter.

3. Process query
    - The system analyzes your question automatically.
    - It searches for relevant context in the document.
    - The model prepares to generate a response.

4. View response
    - Answers appear in a chat-style interface.
    - Responses are generated and streamed in real-time.
    - Relevant document sections are highlighted.

5. Review context
    - See which parts of the document were used.
    - View confidence scores for the answer.
    - Check relevance indicators for accuracy.

6. Continue conversation
    - Ask follow-up questions naturally.
    - Previous context is maintained.
    - View full conversation history.

7. Export interaction
    - Save the Q&A session for later reference.
    - Download highlighted document sections.
    - Export the complete conversation log.

**Additional Features**
- Confidence Scores: Shows how certain the model is about its answer.
- Response Time: Shows the time take by the model to provide a response.
- Contextual Answers: The model uses the context from the document to generate accurate and specific responses.

**Important Notes**
- Accuracy: The answers depend on the model's understanding of the document. If the answer seems off, try rephrasing your question or asking a more specific query.
- Token Limit: The model processes chunks of text from the document. If the document is too long, it may only use parts of it to answer your question.
- Real-Time Streaming: As the model answers, it streams the response, which might make the answer appear in parts. This allows you to read while the response is still being generated.




## Documentation

For detailed documentation, refer to [documentation.md](/documentation.md).