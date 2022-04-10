import matplotlib as plt
from matplotlib import pyplot
from matplotlib.legend import Legend
import  numpy as np
import pandas as pd
import csv

#print(0)
#Accessing the dataset from exteral source
#DATASET FACEBOOK
df1=pd.read_csv("F:/Main Project/Fake profile Detection Usig ML and DL/Datasets/pseudo_facebook.csv")
#DATASET INSTAGRAM
df2=pd.read_csv("F:/Main Project/Fake profile Detection Usig ML and DL/Datasets/users.csv")
#DATASET TWITTER
df3=pd.read_csv("F:/Main Project/Fake profile Detection Usig ML and DL/Datasets/fusers.csv")
###########DATASET facebook ###################
#print(0)

print("\nFacebook Dataset\n")
#Descriptions
print(df1.describe())
#display the coloumn details
print(df1.head)
print(df1)
print(df1.shape)# returns a tuple of the shape of the underlying data for the given series objects
print(f'Total Data Points: {df1.shape[1]}')
print(f'Total columns/Features: {df1.shape[0]}')
##############DATASET Twitter User################################
print("\nTwitter Dataset\n")

#Descriptions
print(df2.describe())
#display the coloumn details
print(df2.head)
print(df2)
print(df2.shape)# returns a tuple of the shape of the underlying data for the given series objects
print(f'Total Data Points: {df2.shape[1]}')
print(f'Total columns/Features: {df2.shape[0]}')
# class distribution
#print(df2.groupby('class').size())


#################DATASET Instagram users ########################################
print("\nInstagram Dataset\n")
# class distribution

#Descriptions
print(df3.describe())
#display the coloumn details
print(df3.head)
print(df3)
print(df3.shape)# returns a tuple of the shape of the underlying data for the given series objects
print(f'Total Data Points: {df3.shape[1]}')
print(f'Total columns/Features: {df3.shape[0]}') 
#######################Plot the Dataset##############################

#####Facebook
df1.plot()
#plt.legend()
pyplot.show()

######Twitter
df2.plot()
pyplot.show()

#######Instagram
df3.plot()
pyplot.show()

#######################Train the model###############################################









