import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from sklearn.datasets import load_iris


import streamlit as st

data = pd.DataFrame(np.random.randint(30, size = (30,3)), columns = ['Col1', 'Col2', 'Col3'])

'Line Graph:'

st.line_chart(data)
#########

'Bar Graph:'

st.bar_chart(data)
###
students = ['A', 'B', 'C']
score = [100,75,25]
attendance = [100,95,90]

x = np.arange(len(score))
width = 0.4

fig, ax = plt.subplots()

ax.bar(x-0.2, score, width, color='blue')
ax.bar(x+0.2, attendance, width, color='black')

ax.set_xticks(x)
ax.set_xticklabels(students)

ax.legend(['Score', 'Attendance'])


st.pyplot(fig)

#########

Classes = ['A', 'B', 'C']

class_count = [100,75,25]

'Pie Chart'

fig, ax = plt.subplots()
ax.pie(class_count, labels= Classes)

st.pyplot(fig)
#########

rows = np.random.randn(1,1)

'Growing Chart:'

chart = st.line_chart(rows)

for i in range(1,100):
    rows = rows[0] + np.random.randn(1,1)
    chart.add_rows(rows)
    #rows = new_
    
    time.sleep(0.05)
#########

iris_data = load_iris()
data = pd.DataFrame(iris_data.data, columns = iris_data.feature_names)

fig = plt.figure()
sns.histplot(data = data, bins = 20)
st.pyplot(fig)

fig = plt.figure()
sns.boxplot(data = data)
st.pyplot(fig)

fig = plt.figure()
sns.scatterplot(data = data)
st.pyplot(fig)

#streamlit run /workspaces/Tutorials-and-Quick-Projects/"Machine Learning"/"Notes & Templates"/1.85_Streamlit.py