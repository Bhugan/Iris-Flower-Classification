import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Streamlit interface
st.title("Iris Flower Species Classification")

st.write("### Upload your CSV file")
uploaded_file = st.file_uploader("Choose a file", type="csv")

if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)
    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]  # Select features
    y = df['species']  # Target variable

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize the KNN model
    knn = KNeighborsClassifier(n_neighbors=3)

    # Train the model using the training data
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    # Calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred)

    # Classification report
    report = classification_report(y_test, y_pred)

    st.write("### Model Accuracy")
    st.write(f"Accuracy: {accuracy * 100:.2f}%")

    st.write("### Classification Report")
    st.text(report)

    st.write("### Predict Iris Flower Species")

    sepal_length = st.number_input("Enter Sepal Length (cm):", min_value=0.0, max_value=10.0, value=5.0)
    sepal_width = st.number_input("Enter Sepal Width (cm):", min_value=0.0, max_value=10.0, value=3.0)
    petal_length = st.number_input("Enter Petal Length (cm):", min_value=0.0, max_value=10.0, value=4.0)
    petal_width = st.number_input("Enter Petal Width (cm):", min_value=0.0, max_value=10.0, value=1.0)

    if st.button("Predict"):
        input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
        prediction = knn.predict(input_data)
        st.write(f"The model predicts the species as: {prediction[0]}")
        st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
