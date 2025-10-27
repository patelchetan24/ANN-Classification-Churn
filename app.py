import tensorflow as tf
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder

# Load the trained model
model=tf.keras.models.load_model("model.h5")

# Load the encoder and scalar
with open("LE.pkl",'rb') as file:
    lable_encoder_gender=pickle.load(file)
with open("one_hot_encoder.pkl","rb") as file:
    lable_encoder_geo=pickle.load(file)
with open("scaler.pkl","rb") as file:
    scalar=pickle.load(file)
    
# streamlit app
st.title("customer churn prediction")

# User input
geography=st.selectbox("Geography",lable_encoder_geo.categories_[0])
gender=st.selectbox("Gender",lable_encoder_gender.classes_)
age=st.slider("Age",18,92)
balance=st.number_input("Balance")
credit_score=st.number_input("Credit Score")
estimated_salary=st.number_input("Estimated Salary")
tenure=st.slider("Tenure",0,10)
num_of_products=st.slider("Number of products",1,4)
has_cr_card=st.selectbox("Has Credit Card",[0,1])
is_active_number=st.selectbox("Is active member",[0,1])
    
# Prepare input data
input_data=pd.DataFrame({
    "CreditScore":[credit_score],
    "Gender":[lable_encoder_gender.transform([gender])[0]],
    "Age":[age],
    "Tenure":[tenure],
    "Balance":[balance],
    "NumOfProducts":[num_of_products],
    "HasCrCard":[has_cr_card],
    "IsActiveMember":[is_active_number],
    "EstimatedSalary":[estimated_salary]
})

# One hot encoded "Geography"
geo_encoded=lable_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=lable_encoder_geo.get_feature_names_out(["Geography"]))

# combine one_hot encoded columns with input data
input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

# Scale the input data
input_data_scaled=scalar.transform(input_data)

# prediction churn
prediction=model.predict(input_data_scaled)
prediction_prob=prediction[0][0]

st.write(f"Churn Probablity:{prediction_prob:.2f}")

if prediction_prob > 0.5:
    st.write("The customer is likely to churn")
else:
    st.write("The customer is not likely to churn")
    