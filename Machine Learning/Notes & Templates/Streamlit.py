# streamlit run ./Streamlit.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import streamlit as st

st.title('Model Selection',)

st.divider()

st.code("""
def function:
        print('hello')
        return

""", language="python")
#Adding Image
#w = 100
#st.image(os.path.join(os.getcwd(), 'Static', 'Image.jpg'), width = w)

x = np.array([[1,2,3],[4,5,6]])

y = pd.DataFrame({'Name':['A', 'b', 'C'],
                  'Age':[1,2,3]})
z = pd.DataFrame({'Name':[1],
                  'Age':[1]})
st.subheader('Editor')
z_ =st.data_editor(z)

def show():
    st.subheader('Y')
    st.dataframe(y)    
def reset():
    st.subheader('Z')
    st.dataframe(z)


def edited():
    st.subheader('Edited')
    st.table(z_)

button = st.button('Press Me', on_click=show)

button = st.button('Reset', on_click=reset)

button = st.button('Show Edited', on_click=edited)

st.button("Reset", type="primary")

if st.button("Say hello"):
    st.write("Why hello there")
else:
    st.write("Goodbye")

if st.button("Aloha", type="tertiary"):
    st.write("Ciao")