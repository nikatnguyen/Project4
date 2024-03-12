#Dependencies
import streamlit as st
import pickle
import pandas as pd
import sklearn as skl
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#Load original data
original_data = pd.read_csv('https://raw.githubusercontent.com/nikatnguyen/Project4/main/Resources/ObesityDataSet.csv')
original_data = original_data.drop(columns = ['NCP', 'SCC', 'TUE'])


# Load the saved model
model_path = 'final_model.pkl'

# Streamlit App
def main():

    st.title("Predicting Obesity Using Machine Learning Model")
    st.write("""This app allows you to predict an obesity diagnostic using a series of questions based on a dataset taken from Kaggle
    """)

    # Sidebar with user input
    st.sidebar.header("Questionnaire")

    #Defining options for checkboxes
    yn_options = ["Yes", "No"]
    tf_options = ["True", "False"]
    frequency_options = ["Frequently", "Sometimes", "no"]
    transport_options = ["Bike", "Motobike", "Public Transportation", "Waking", "Automobile"]

    # Example input features (you can replace this with your actual input fields)
    age = st.sidebar.number_input("How old are you?", 0.0, 100.0, 50.0)
    height = st.sidebar.number_input("What is your height in meters?", 0.0, 2.0)
    weight = st.sidebar.number_input("What is your weight in kilograms?", 0.0, 200.0, 50.0)
    
    favc_question = st.sidebar("Do you frequently consume high caloric foods? (Frequently meaning 3 or more times a day)")
    favc_option_y = st.sidebar.checkbox("Yes")
      if favc_option_y:
       favc = 1
    favc_option_n = st.sidebar.checkbox("No")
      if favc_option_n:
       favc = 0
    
    fcvc = st.sidebar.select_slider("How frequently do you consume vegetables each day? (3 meaning at more than 2 servings a day, 2 meaning around 2 servings a day, 1 meaning one serving a day, 0 being none)", options=[0, 1, 2, 3])
    ncp = st.sidebar.number_input("How many main meals do you generally have each day?", 0, 4, 2)
    
    caec_question = st.sidebar("How often do you consume food in between meals?")
    caec_frequently = st.sidebar.checkbox("Frequently")
    caec_sometimes = st.sidebar.checkbox("Sometimes")
    caec_no = st.sidebar.checkbox("Never")
    
    ch20_question = st.sidebar("How often do you drink water each day?")
    ch20_frequently = st.sidebar.checkbox("Frequently")
      if ch20_frequently: 
       ch20 = "Frequently"
    ch20_sometimes = st.sidebar.checkbox("Sometimes")
      if ch20_sometimes:
        ch20 = "Sometimes"
    ch20_no = st.sidebar.checkbox("Never")
      if ch20_no:
        ch20 = "No"

    faf = st.sidebar.number_input("How frequently do you exercise each day?", 0, 3, 2)
    ch2o = st.sidebar.select_slider("How often do you drink water each day? (3 is frequently or about 6-8 cups a day, 2 is less frequently meaning 3-5 cups a day, 1 is 1-2 cups a day, and 0 is none)", options=[0, 1, 2, 3])
    
    calc_question = st.sidebar("Do you frequently consume alcohol? (Frequently meaning 3 or more glases a day)")
    calc_frequently = st.sidebar.checkbox("Frequently")
    calc_sometimes = st.sidebar.checkbox("Sometimes")
    calc_no = st.sidebar.checkbox("Never")

    smoke_question = st.sidebar("Do you smoke?")
    smoke_yes = st.sidebar.checkbox("Yes")
      if smoke_yes:
       smoke = 1
    smoke_no = st.sidebar.checkbox("No")
      if smoke_no:
        smoke = 0
    
    transportation_question = st.sidebar("What mode of transportation do you generally take each day?", options=transport_options)
    mtrans_bike = st.sidebar.checkbox("Bike")
    mtrans_motorbike = st.sidebar.checkbox("Motorbike")
    mtrans_public = st.sidebar.checkbox("Public Transportation")
    mtrans_walk = st.sidebar.checkbox("Walking")
    mtrans_other = st.sidebar.checkbox("Other")

    #Load model
    with open(model_path, 'rb') as model_file:
        loaded_model = pickle.load(model_file)

    # Make predictions with the loaded model
    st.form_submit_button("Predit")
      if st.form_submit_button("Predict"):
        user_data = pd.DataFrame({
          'Age': [age],
          'Height': [height],
          'Weight': [weight],
          'FCVC': [fcvc],
          'CH20': [ch20],
          'FAF': [faf],
          'FAVC_yes': [favc],
          "CAEC_Frequently": [caec_frequently],
          "CALC_Sometimes":  [caec_sometimes],
          "CAEC_no": [caec_no],
          "SMOKE_yes":  [smoke],
          "CALC_Frequently":  [calc_frequently],
          "CALC_Sometimes":  [calc_sometimes],
          "CALC_no":  [calc_no],
          "MTRANS_Bike":  [mtrans_bike],
          "MTRANS_Motorbike":  [mtrans_motorbike],
          "MTRANS_Public_Transportation":  [mtrans_public],
          "MTRANS_Walking":  [mtrans_walk] })
        combined_df = pd.concat([original_data, user_data], axis = 0)
        combined_df = pd.get_dummies(combined_df, drop_first = True)
        user_data = combined_df.iloc[-1, :]
        prediction = loaded_model.predict(user_data)
        # Display the prediction
        st.subheader("Prediction:")
        st.write(prediction)

    st.write("""Dataset from Kaggle: https://www.kaggle.com/datasets/aravindpcoder/obesity-or-cvd-risk-classifyregressorcluster
    """)
if __name__ == "__main__":
    main()

