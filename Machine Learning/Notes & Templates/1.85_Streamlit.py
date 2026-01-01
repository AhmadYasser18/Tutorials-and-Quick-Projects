import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import streamlit as st

data = pd.DataFrame(np.random.randint(30, size = (30,3)), columns = ['Col1', 'Col2', 'Col3'])

'Line Graph:'

st.line_chart(data)
#########

'Bar Graph:'

st.bar_chart(data)
#########

Classes = ['A', 'B', 'C']

class_count = [100,75,25]

'Pie Chart'

fig, ax = plt.subplots()
ax.pie(class_count, labels= Classes)

st.pyplot(fig)
#########
#streamlit run /workspaces/Tutorials-and-Quick-Projects/"Machine Learning"/"Notes & Templates"/1.85_Streamlit.py