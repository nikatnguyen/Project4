# Project 4 (Group 2): Predicting BMI Ranges

ABOUT OUR DATASET
Sourced from Kaggle, our dataset holds the estimation of obesity levels in people from the countries of Mexico, Peru, and Colombia; Ages ranging between 14 and 61 with diverse eating habits and physical condition. The data was collected anonymously using a web platform, completing with 17 columns and 2111 records.

OUR GOAL
We sought to develop a Machine Learning model designed to evaluate an individual's physical health and identify the specific range of the BMI Range they fall within. By deploying this model onto a user-friendly Streamlit dashboard, we facilitate a seamless experience for users who can receive personalized predictions by answering a set of questions. Our overarching objective is to contribute to proactive health management by offering an accessible tool that provides a general picture of user’s health as well as medical resources.

PREPROCESSING
Columns we decided to drop were the number of main meals and whether calorie intake or technology use is monitored, as the contexts of these are too subjective of an individuals needs for our model.
We converted binary and frequency columns like sex, family history, frequency eating high-caloric foods, eating between meals, smoking, alcohol consumption, and transportation types into separate integer columns. Each column is answered in a 1[Yes] or 0[no]
Our target was the 'NObesydad' column in the original training dataset, and our features were everything else.

BUILDING OUR MODEL
Our earliest model, Michael used the standard optimization method for my model and tested across three configurations. He experimented with different numbers of layers, units, and epochs in accordance with the shape of the model.
Upon testing, the first model had an accuracy score of 0.92; The other models gave lackluster results, dropping in value with over-fitting and overly complex neuron values. We needed a less complex modeling approach.

Gonzalo opted for using the K-fold cross validation to find optimal scores, which resulted in an average fold accuracy score of 94%. The optimal model that was found was the RandomForestClassifier model, which was used in Anika's optimization script in building the model. 

We chose Streamlit off of recommendations, but it really is an easy to use platform that made our deployment process (overall) smooth. The app is basically designed to take in user data the same way it’s taken in the original training dataset. Then the user answers are saved into a data frame and concatenated the user input with the original dataset. Finally, to get the predictions we applied the get_dummies method. 

APP LIMITATIONS
-Our model is unable to calculate actual BMI scores, but it is meant to provide users a general picture of their physical health based on their lifestyle choices.
-Our app utilizes the metric system over the imperial system based on our original dataset
-Selecting multiple checkboxes under a question will result in an error
-In building the model for the app, our n_estimators and max_depth ranges were limited, and increasing the range of these may result in a higher accuracy than our final model. 

OUR APP: https://project4-obesitypredictionapp.streamlit.app/

FILES IN REPO

SQL DATABASE 
- obesity_dummies.db
  
PREPROCESSING/OPTIMIZATION FILES
-Preprocessing.ipynb
-Mj_optimization.ipynb
-Project4_GonzaloAmbriz.ipynb
-anika_optimization.ipynb

APP/PKL FILES
-app.py
-final_model.pkl
-requirements.txt
-scaler.pkl

DATASETS
-Resources/ObesityDataSet.csv (original from kaggle)
-Resources/obesity_dummies_df.csv


