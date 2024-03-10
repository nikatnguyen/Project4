import numpy as numpy
import pandas as pandasimport streamlit as st
from sklearn import preprocessing
import pickle

model = pickle.load(open('mj_obesity_nn.pkl', 'rb'))
encoder_dict = pickle.load(open('encoder.pkl', 'rb'))