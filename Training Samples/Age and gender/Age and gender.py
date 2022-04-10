#importing libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
# plotting
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

######################
# Image processing
from PIL import Image

##############

#Adding csv file 

df=pd.read_csv("F:/Main Project/MCA26/Age and gender/Dataset/age_gender.csv")

print(df.head())#returns the first n rows(observe the index values).
#tail: displays the last five rows of the dataframe by default

print(df.shape)# returns a tuple of the shape of the underlying data for the given series objects
print(f'Total Data Points: {df.shape[1]}')
print(f'Total columns/Features: {df.shape[0]}')

df.info()#Displays all related features of the documents

print(type(df.pixels[0]) )# since pixels are in form of string we need to convert it to an array

## Converting pixels into numpy array
df['pixels'] = df['pixels'].apply(lambda x:  np.reshape(np.array(x.split(), dtype="float32"), (48,48)))
print(df.head())

print(type(df.pixels[0]) )

fig= px.histogram(df, x="age")
#print(df)
fig.update_layout(title_text="Age Histogram")
#print(fig.update_layout(title_text="Age Histogram"))
fig.show()
#print(plt.show())

#########################
###Labeling
eth_values_to_labels = { 0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Hispanic" }
gender_values_to_labels = { 0: "Male", 1: "Female" }

##########################
print(df.ethnicity.value_counts())

#plot the ethnicity value
fig = go.Figure([
    go.Bar(x=[eth_values_to_labels[i] for i in df.ethnicity.value_counts().index], 
           y=df.ethnicity.value_counts().values)
])
fig.update_layout(
    title_text='Count Plot Ethnicity',
    xaxis_title='Ethnicity',
    yaxis_title='Count'
)
fig.show()



#Extract the gender count 
print(df.gender.value_counts())

