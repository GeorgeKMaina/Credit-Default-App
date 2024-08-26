# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 09:23:24 2024

@author: Gmaina
"""

import numpy as np
import pandas as pd
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

def main():
    st.title("Credit Default Prediction App")
    st.write("""
    This application predicts whether a customer will default on their credit based on the provided information.
    Please fill in the details below and click **Predict** to see the result.
    """)
    
    # Load the trained model and encoder
    @st.cache_resource
    def load_model():
        model_path = r"C:\users\gmaina\Downloads\trained_model.sav"
        encoder_path = r"C:\users\gmaina\encoder.sav"
        with open(model_path, 'rb') as model_file:
            loaded_model = pickle.load(model_file)
        with open(encoder_path, 'rb') as encoder_file:
            encoder = pickle.load(encoder_file)
        return loaded_model, encoder
    
    loaded_model, encoder = load_model()
    
    # Define selection options
    months_of_year = (
        "January", "February", "March", "April", "May", "June", 
        "July", "August", "September", "October", "November", "December"
    )
    
    days_of_week = (
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    )
    
    AccidentArea = ("Urban", "Rural")
    Sex = ("Male", "Female")
    Marital_status = ('Single', 'Married', 'Widow', 'Divorced')
    Policy_type = ('Policy Holder', 'Third Party')
    VehicleCategory = ('Sport', 'Sedan', 'Utility')
    Days_Policy_Accident = ('more than 30', '15 to 30', 'none', '1 to 7', '8 to 15')
    Days_Policy_Claim = ('more than 30', '15 to 30', '8 to 15')
    PoliceReportFiled = ('No', 'Yes')
    WitnessPresent = ('No', 'Yes')
    AgentType = ('External', 'Internal')
    AddressChange_Claim = ('1 year', 'no change', '4 to 8 years', '2 to 3 years', 'under 6 months')
    NumberOfCars = ('1 vehicle', '2 vehicles', '3 to 4', '5 to 8', 'more than 8')
    VehiclePrice = ('less than 20k', '20k to 30k', '30k to 40k', '40k to 50k', 'more than 50k')
    AgeOfVehicle = ('new', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years or more')
    Deductible = ('500', '1000', '1500', '2000')
    DriverRating = ('1', '2', '3', '4', '5')
    PastNumberOfClaims = ('0', '1', '2', '3', '4 or more')
    AgeOfPolicyHolder = ('18-25', '26-35', '36-45', '46-55', '56-65', '66 or older')
    NumberOfSuppliments = ('0', '1', '2', '3', '4 or more')
    Make = (
        'Honda', 'Toyota', 'Ford', 'Mazda', 'Chevrolet', 'Pontiac',
        'Acura', 'Dodge', 'Mercury', 'Jaguar', 'Nissan', 'VW', 
        'Saab', 'Saturn', 'Porsche', 'BMW', 'Mercedes', 'Ferrari', 'Lexus'
    )
    
    # Collect user inputs
    st.subheader("Policy Information")
    PolicyNumber = st.text_input("Customer Policy Number")
    Policy_type_selected = st.selectbox('Policy Type', Policy_type)
    Days_Policy_Accident_selected = st.selectbox('Days Since Policy Accident', Days_Policy_Accident)
    Days_Policy_Claim_selected = st.selectbox('Days Since Policy Claim', Days_Policy_Claim)
    AddressChange_Claim_selected = st.selectbox('Address Change Since Claim', AddressChange_Claim)
    AgentType_selected = st.selectbox('Agent Type', AgentType)
    
    st.subheader("Vehicle Information")
    Make_selected = st.selectbox('Vehicle Make', Make)
    VehicleCategory_selected = st.selectbox('Vehicle Category', VehicleCategory)
    VehiclePrice_selected = st.selectbox('Vehicle Price', VehiclePrice)
    AgeOfVehicle_selected = st.selectbox('Age of Vehicle', AgeOfVehicle)
    NumberOfCars_selected = st.selectbox('Number of Vehicles Owned', NumberOfCars)
    Deductible_selected = st.selectbox('Deductible Amount', Deductible)
    DriverRating_selected = st.selectbox('Driver Rating', DriverRating)
    
    st.subheader("Accident Information")
    AccidentArea_selected = st.selectbox('Accident Area', AccidentArea)
    PoliceReportFiled_selected = st.selectbox('Police Report Filed', PoliceReportFiled)
    WitnessPresent_selected = st.selectbox('Witness Present', WitnessPresent)
    Month_selected = st.selectbox('Month of Accident', months_of_year)
    Weekday_selected = st.selectbox('Day of Accident', days_of_week)
    
    st.subheader("Customer Information")
    Sex_selected = st.selectbox('Gender', Sex)
    Marital_status_selected = st.selectbox('Marital Status', Marital_status)
    AgeOfPolicyHolder_selected = st.selectbox('Age of Policy Holder', AgeOfPolicyHolder)
    PastNumberOfClaims_selected = st.selectbox('Past Number of Claims', PastNumberOfClaims)
    NumberOfSuppliments_selected = st.selectbox('Number of Supplements', NumberOfSuppliments)
    
    # Prepare input data for prediction
    input_data = {
        'PolicyNumber': PolicyNumber,
        'PolicyType': Policy_type_selected,
        'DaysPolicyAccident': Days_Policy_Accident_selected,
        'DaysPolicyClaim': Days_Policy_Claim_selected,
        'AddressChangeClaim': AddressChange_Claim_selected,
        'AgentType': AgentType_selected,
        'VehicleMake': Make_selected,
        'VehicleCategory': VehicleCategory_selected,
        'VehiclePrice': VehiclePrice_selected,
        'AgeOfVehicle': AgeOfVehicle_selected,
        'NumberOfVehicles': NumberOfCars_selected,
        'Deductible': Deductible_selected,
        'DriverRating': DriverRating_selected,
        'AccidentArea': AccidentArea_selected,
        'PoliceReportFiled': PoliceReportFiled_selected,
        'WitnessPresent': WitnessPresent_selected,
        'Month': Month_selected,
        'Weekday': Weekday_selected,
        'Gender': Sex_selected,
        'MaritalStatus': Marital_status_selected,
        'AgeOfPolicyHolder': AgeOfPolicyHolder_selected,
        'PastNumberOfClaims': PastNumberOfClaims_selected,
        'NumberOfSuppliments': NumberOfSuppliments_selected
    }
    
    input_df = pd.DataFrame([input_data])
    
    # Encoding categorical variables
    try:
        input_df_encoded = encoder.transform(input_df)
    except Exception as e:
        st.error(f"Error in encoding input data: {e}")
        return
    
    # Make prediction
    if st.button("Predict"):
        prediction = loaded_model.predict(input_df_encoded)
        prediction_proba = loaded_model.predict_proba(input_df_encoded)
        
        if prediction[0] == 1:
            result = 'Default'
            confidence = prediction_proba[0][1] * 100
        else:
            result = 'No Default'
            confidence = prediction_proba[0][0] * 100
        
        st.success(f"Prediction: **{result}**")
        st.info(f"Confidence: **{confidence:.2f}%**")
    
    st.write("---")
    st.write("Developed by [Your Name]")

if __name__ == '__main__':
    main()
