import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


iris = load_iris()
X = iris.data
Y = iris.target


rf_classifier = RandomForestClassifier()
rf_classifier.fit(X, Y)

st.title("Iris Flower Prediction")
st.header("Enter the following parameters:")


sepal_length = st.slider("Sepal Length", float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
sepal_width = st.slider("Sepal Width", float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
petal_length = st.slider("Petal Length", float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
petal_width = st.slider("Petal Width", float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))

if st.button("Predict"):

    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = rf_classifier.predict(input_data)


    predicted_species = iris.target_names[prediction[0]]
    st.write(f"Predicted Iris Species: {predicted_species}")