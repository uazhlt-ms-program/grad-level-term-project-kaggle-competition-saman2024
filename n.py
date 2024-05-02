import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load the training dataset from a CSV file
data = pd.read_csv('train.csv')

# Handle missing values in the 'TEXT' column by dropping rows with missing values
# and filling NaN values with empty strings
data.dropna(subset=['TEXT'], inplace=True)
data['TEXT'].fillna('', inplace=True)

# Split the data into training and validation sets using train_test_split function
# with 80% of the data for training and 20% for validation, ensuring reproducibility
X_train, X_val, y_train, y_val = train_test_split(data['TEXT'], data['LABEL'], test_size=0.2, random_state=42)

# Define a pipeline for text classification consisting of two main components:
# - TfidfVectorizer for converting text data into TF-IDF numerical representation
# - LogisticRegression classifier for label prediction
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),  # TF-IDF representation
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))  # Logistic Regression
])

# Fit the pipeline to the training data, training both the vectorizer and the classifier
pipeline.fit(data['TEXT'], data['LABEL'])

# Load the test data from a CSV file
test_data = pd.read_csv('test.csv')

# Handle missing values in the 'TEXT' column of the test data by filling NaN values with empty strings
test_data['TEXT'].fillna('', inplace=True)

# Predict labels for the test data using the trained pipeline
predictions = pipeline.predict(test_data['TEXT'])

# Create a DataFrame containing the predicted labels along with their corresponding IDs
submission_df = pd.DataFrame({'ID': test_data['ID'], 'LABEL': predictions})

# Save the predictions to a CSV file named 'submission99.csv' without including the index column
submission_df.to_csv('submission99.csv', index=False)
