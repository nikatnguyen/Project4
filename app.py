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
original_data = original_data.drop(columns = ['NCP', 'SCC', 'TUE', 'NObeyesdad'])


# Load the saved models
fin_model_path = 'final_model.pkl'
scaler_path = 'scaler.pkl'

# Streamlit App
def main():

    st.title("Predicting BMI Indexes Using Machine Learning Model")
    st.write("""This app allows you to predict your BMI index using a series of questions based on a dataset taken from Kaggle surveying people ages 14-61 from Mexico, Peru, and Columbia with diverse eating habits and physical conditions. It does not calculate the actual BMI score of users but is designed to give users an idea of their physical health  
    """)
    st.write("""DISCLAIMER: This app is designed for users concerned about their physical health, but it should not be taken as a formal medical diagnosis of an individual's health. Consult your doctor or see linked resources at the bottom of this questionnaire for more information""")

    # Sidebar with user input
    st.header("Questions")
    st.write("""Selecting multiple checkboxes in each question will result in an error.""")

    
    # Example input features (you can replace this with your actual input fields)
    gender_question = st.write("What is your sex assigned at birth?")
    male = st.checkbox("Male", key="male")
    if male:
      gender = "Male"
    female = st.checkbox("Female", key="fem")
    if female:
      gender = "Female"
    age = st.number_input("How old are you?", 0.0, 100.0, 50.0)
    height = st.number_input("What is your height in meters?", 0.0, 2.0)
    weight = st.number_input("What is your weight in kilograms?", 0.0, 200.0, 50.0)
    
    favc_question = st.write("Do you frequently consume high caloric foods? (Frequently meaning 3 or more times a day)")
    favc_option_y = st.checkbox("Yes", key="favcy")
    if favc_option_y:
      favc = "yes"
    favc_option_n = st.checkbox("No", key="favcn")
    if favc_option_n:
      favc = "no"
    
    fcvc = st.select_slider("How frequently do you consume vegetables each day? (3 meaning at more than 2 servings a day, 2 meaning around 2 servings a day, 1 meaning one serving a day, 0 being none)", options=[0, 1, 2, 3])
    
    caec_question = st.write("How often do you consume food in between meals?")
    caec_always = st.checkbox("Always", key="caeca")
    if caec_always:
       caec = "Always"                         
    caec_frequently = st.checkbox("Frequently", key="caecf")
    if caec_frequently:
       caec = "Frequently"
    caec_sometimes = st.checkbox("Sometimes", key="caecsome")
    if caec_sometimes:
       caec = "Sometimes"
    caec_no = st.checkbox("Never", key="caecno")
    if caec_no:
       caec = "no"
    

    faf = st.number_input("How frequently do you exercise each day?", 0, 3, 2)
    ch2o = st.select_slider("How often do you drink water each day? (3 is frequently or about 6-8 cups a day, 2 is less frequently meaning 3-5 cups a day, 1 is 1-2 cups a day, and 0 is none)", options=[0, 1, 2, 3])
    
    family_question = st.write("Do you have a family history with obesity?")
    family_true = st.checkbox("Yes", key="famtrue")
    if family_true:
      family = "yes"
    family_false = st.checkbox("No", key="famfalse")
    if family_false:
       family = "no"
    family_unsure = st.checkbox("Unsure", key="famunsure")
    if family_unsure:
       family = "no"
    
    calc_question = st.write("Do you frequently consume alcohol? (Frequently meaning 3 or more glases a day)")
    calc_frequently = st.checkbox("Frequently", key="calcfreq")
    if calc_frequently:
       calc = "Frequently"
    calc_sometimes = st.checkbox("Sometimes", key="caclsome")
    if calc_sometimes:
       calc = "Sometimes"
    calc_no = st.checkbox("Never", key="calcno")
    if calc_no:
       calc = "no"

    smoke_question = st.write("Do you smoke?")
    smoke_yes = st.checkbox("Yes", key="smokey")
    if smoke_yes:
      smoke = "yes"
    smoke_no = st.checkbox("No", key="smokeno")
    if smoke_no:
      smoke = "no"
    
    transportation_question = st.write("What mode of transportation do you generally take each day?")
    mtrans_bike = st.checkbox("Bike", key="bike")
    if mtrans_bike:
       mtrans = "Bike"
    mtrans_motorbike = st.checkbox("Motorbike", key="motorbike")
    if mtrans_motorbike:
       mtrans = "Motorbike"
    mtrans_public = st.checkbox("Public Transportation", key="publictrans")
    if mtrans_public:
       mtrans = "Public_Transportation"
    mtrans_walk = st.checkbox("Walking", key="walk")
    if mtrans_walk:
       mtrans = "Walking"
    mtrans_car = st.checkbox("Automobile", key="car")
    if mtrans_car:
       mtrans = "Automobile"

    #Load model
    with open(fin_model_path, 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    with open(scaler_path, 'rb') as model_file:
        X_scaler = pickle.load(model_file)
    # Make predictions with the loaded model
    if st.button("Predict"):
      user_data = pd.DataFrame({
        'Gender': [gender],
        'Age': [age],
        'Height': [height],
        'Weight': [weight],
        'family_history_with_overweight': [family],
        'FAVC': [favc],
        'FCVC': [fcvc],
        'CAEC': [caec],
        'SMOKE': [smoke],
        'CH2O': [ch2o],
        'FAF': [faf],
        'CALC': [calc],
        "MTRANS": [mtrans]})
      combined_df = pd.concat([original_data, user_data], axis = 0)
      combined_df = pd.get_dummies(combined_df, columns = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'CALC', 'MTRANS'], drop_first = True)
      user_data = combined_df.iloc[-1, :]
      user_data = pd.DataFrame(user_data.values.reshape(1, -1), columns=combined_df.columns)
      user_data = X_scaler.transform(user_data)
      prediction = loaded_model.predict(user_data)
      # Display the prediction
      st.subheader("Prediction:")
      if prediction == 0:
        st.write("Underweight")
        st.write("""Based on your results, your BMI index seems to fall below 18.5""")
      if prediction == 1:
        st.write("Normal weight")
        st.write("""Based on your results, your BMI index seems to fall within the range of 18.5-24.9""")
      if prediction == 2:
        st.write("Obesity Type 1")
        st.write("""Based on your results, your BMI index seems to be above 30""")
      if prediction == 3:
        st.write("Obesity Type 2")
        st.write("""Based on your results, your BMI index seems to be above 30""")
      if prediction == 4:
        st.write("Obesity Type 3")
        st.write("""Based on your results, your BMI index seems to be above 30""")
      if prediction == 5:
        st.write("Overweight Level 1")
        st.write("""Based on your results, your BMI index seems to fall within the range of 25.0-29.9""")
      if prediction == 6:
        st.write("Overweight Level 2")
        st.write("""Based on your results, your BMI index seems to fall within the range of 25.0-29.9""")

    st.header("Resources For More Information")
    st.markdown('- [About Adult BMI (CDC)](https://www.cdc.gov/healthyweight/assessing/bmi/adult_bmi/index.html)', unsafe_allow_html=True)
    st.markdown('- [Body Mass Index (CDC)](https://www.cdc.gov/healthyweight/assessing/bmi/index.html#:~:text=Calculate%20Your%20BMI&text=Body%20Mass%20Index%20(BMI)%20is,or%20health%20of%20an%20individual', unsafe_allow_html=True)
    st.markdown('- [The Impact of Obesity on Your Body (American Society for Metabolic and Bariatric Surgery)](https://asmbs.org/patients/impact-of-obesity/)', unsafe_allow_html=True)
    st.markdown('- [Understanding Obesity (Obesity Action Coalition)](https://www.obesityaction.org/get-educated/public-resources/brochures-guides/understanding-obesity-brochure/)', unsafe_allow_html=True)
    st.markdown('- [Why Does Weight Bias Matter (Obesity Care Week)](https://www.obesitycareweek.org/wp-content/uploads/2023/02/OCW-Why-Does-Weight-Bias-Matter-1.pdf)', unsafe_allow_html=True)
    st.markdown('- [Access to Care (Obesity Care Week)](https://www.obesitycareweek.org/wp-content/uploads/2023/02/OCW-Access-to-Care.pdf)', unsafe_allow_html=True)
    st.markdown('- [The Benefits of Treatment (Obesity Care Week)](https://www.obesitycareweek.org/wp-content/uploads/2023/02/OCW-Benefits-of-Treatment-1.pdf)', unsafe_allow_html=True)
    st.markdown('- [Obesity Prevention Strategies and Guidelines (CDC)](https://www.cdc.gov/obesity/resources/strategies-guidelines.html)', unsafe_allow_html=True)
    
    st.write("""Dataset from Kaggle: https://www.kaggle.com/datasets/aravindpcoder/obesity-or-cvd-risk-classifyregressorcluster
    """) 
if __name__ == "__main__":
    main()

