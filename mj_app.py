import numpy as numpy
import pandas as pandasimport streamlit as st
from sklearn import preprocessing
import pickle

model = pickle.load(open('mj_obesity_nn.pkl', 'rb'))
encoder_dict = pickle.load(open('encoder.pkl', 'rb'))
cols=['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'Insufficient_Weight',
    'Normal_Weight', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III',
    'Overweight_Level_I','Overweight_Level_II', 'Female', 'Male', 'no', 'yes', 'no', 'yes', 'Always',
    'Frequently', 'Sometimes', 'no', 'no', 'yes', 'no', 'yes', 'Always', 'Frequently', 'Sometimes', 'no',
    'Automobile', 'Bike', 'Motorbike', 'Public_Transportation', 'Walking']

def main():
    st.title("Obese Predictor")
    html_temp = """
    <div style="background:#025246; padding:10px">
    <h2 style="color:white;text-align:center;">Obesity Prediction App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)

    age = st.text_input("Age", "0")
    height = st.text_input("Height", "0.0")
    weight = st.text_input("Weight", "0")
    sex = st.selectbox("Sex",["Male", "Female"])

    if st.button("Predict"):
        features = [[age,height,weight,sex]]
        data = {'age': int(age), 'height': float(height), 'weight': int(weight), 'sex': sex}
        print(data)
        df=pd.DataFrame([list(data.values())], columns= ['age', 'height', 'weight', 'sex'])

        category_col = ['sex']
        for cat in encoder_dict:
            for col in df.columns:
                le = preprocessing.LabelEncoder()
                if cat == col:
                    le.classes_ = encoder_dict[cat]
                    for unique_item in df[col].unique():
                        if unique_item not in le.classes_:
                            df[col] = ['Unknown' if x == unique_item or else x for x in df[col]]
                    df[col] = le.transform(df[col])

        features_list = df.values.tolist()
        prediction = model.predict(features_list)

        output = int(prediction[0])
        if output == 1:
            text = "Obese!"
        else:
            text = "Not obese!"
        
        st.success('')