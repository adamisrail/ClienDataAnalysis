import numpy as np
import pandas as pd

# HEADING: Importing the dataset
datasetPath = r'Data/Sample.xlsx' #Locating the dataset
dataset = pd.read_excel(datasetPath) #Reading the excel using pandas
print(dataset.columns.ravel()) #Viewing colums of the frame
type(dataset) #Checking the type of dataset

#HEADING: Pre-Processing
#SUB-HEADING: Analyzing the data to check if all datatypes have been read accurate

dataset.info() #Checking if the datatypes of columns are okay
pd.set_option('display.max_columns', None) #So that when use pandas head function, columns are not hidden
dataset.head() #Checking if the data is either cutout, for example the columns having value starting with 0 might loose the 0

#Aquired data was read as object hence changing it to date time
dataset['acquired_date'] = pd.to_datetime(dataset['acquired_date'], format='mixed') #Using format = mixed since the format is mixed
dataset['acquired_date'].head(5)

dataset.info() ##Transaction time is not set to UTC
dataset['transaction_time'] = pd.to_datetime(dataset.transaction_time).dt.tz_localize('UTC') #Setting time to UTC
dataset.info() #Looks Good

#Now changing the product type to object as well since this is supposed to be an replacement for name of product, not an int
dataset['product_type'] = dataset['product_type'].astype('str')
dataset.info() #Checking if the datatypes of columns are okay
#Datatypes look good for now! Might change them in future if required


#SUB-HEADING: Analyzing the data to find missing values

dataset.isna().sum() #Looking for missing values of each feature
dataset.isna().sum().sum() #Total missing values
#No missing values found

print(dataset[dataset == np.inf].count() + dataset[dataset == -np.inf].count()) #Total infinity values per each feature
print((dataset[dataset == np.inf].count() + dataset[dataset == -np.inf].count()).sum()) #Total infinity values present
#No infinity values found

#SUB-HEADING: Analyzing the data to find outliers
import scipy.stats as stats
statsData = stats.zscore(dataset['value']) #Looking at the Z-Scores of feature value

#Visualizing the feature
import matplotlib.pyplot as plt

#creating a box plot of feature value
plt.boxplot(dataset['value'])

#creating a scatter plot with x axis being constant 1 for viewing
plt.xlabel("Constant 1 just for viewing")
plt.ylabel("Value of dataset['value']")
plt.scatter(np.ones((len(dataset['value']))), dataset['value'], s=1) #setting the size of dot small to have a better visual

#creating a scatter plot with z-scores and value just for different views
plt.xlabel("z-score of dataset['value']")
plt.ylabel("Value of dataset['value']")
plt.scatter(statsData, dataset['value'], s=1) #setting the size of dot small to have a better visual

#creating a histogram with 50 bins to view outliers
plt.hist(dataset['value'], bins=50)

#It can be seen that outliers exist in the data. Hence removing the outliers.
#NOTE: All these outliers are based on a positive z score which means high valued clients
#Storing them to as they are a good fit to target!

from scipy.stats import zscore
dataset['z_score'] = zscore(dataset['value'])
dataset.info() #Checking if the datatypes of columns are okay
threshold = 3
df_exceeding_threshold = dataset[dataset['z_score'] > threshold]
df_exceeding_threshold.info()
df_exceeding_threshold







