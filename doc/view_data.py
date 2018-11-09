# -*- coding: utf-8 -*-

""" This module computed statistics and described the forest cover type dataset.
    Warning: be careful when running this code: huge amount of images.
    Reference: https://www.kaggle.com/sharmasanthosh/exploratory-study-on-feature-selection
"""

import warnings
import pandas
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Read raw data from the file
dataset = pandas.read_csv("../data/train-data.csv")
dataset = dataset.iloc[:,1:]

""" 1. Size of the dataframe """

print(dataset.shape)

# (15120, 55)
# There are 15120 instances having 55 attributes

""" 2. Datatypes of the attributes """
print(dataset.dtypes)

# Data types of all attributes has been inferred as int64

""" 3. Statistical description """
pandas.set_option('display.max_columns', None)
print(dataset.describe())

# No attribute is missing as count is 15120 for all attributes. Hence, all rows can be used
# Negative value(s) present in Vertical_Distance_To_Hydrology. Hence, some tests such as
# chi-sq cant be used. Wilderness_Area and Soil_Type are one hot encoded. Hence, they could
# be converted back for some analysis Attributes Soil_Type7 and Soil_Type15 can be removed
# as they are constant Scales are not the same for all. Hence, rescaling and standardization
# may be necessary for some algos

""" 4. Skewness of the distribution """

print(dataset.skew())

# Values close to 0 show less skew
# Several attributes in Soil_Type show a large skew. Hence, some algos may benefit if skew is corrected

""" 5. Class distribution """
# Number of instances belonging to each class

print(dataset.groupby('Cover_Type').size())

# All classes have an equal presence. No class re-balancing is necessary

""" 6. Correlation """
# Relation between two attributes.
# Correlation requires continous data.
# Hence, ignore data: Wilderness_Area, Soil_Type and Cover_type

#sets the number of features considered
size = 10

#create a dataframe with only 'size' features
data=dataset.iloc[:,:size]

#get the names of all the columns
cols=data.columns

# Calculates pearson co-efficient for all combinations
data_corr = data.corr()

# Set the threshold to select only only highly correlated attributes
threshold = 0.5

# List of pairs along with correlation above threshold
corr_list = []

#Search for the highly correlated pairs
for i in range(0,size): #for 'size' features
    for j in range(i+1,size): #avoid repetition
        if (data_corr.iloc[i,j] >= threshold and data_corr.iloc[i,j] < 1) or (data_corr.iloc[i,j] < 0 and data_corr.iloc[i,j] <= -threshold):
            corr_list.append([data_corr.iloc[i,j],i,j]) #store correlation and columns index

#Sort to show higher ones first
s_corr_list = sorted(corr_list,key=lambda x: -abs(x[0]))

#Print correlations and column names
for v,i,j in s_corr_list:
    print ("%s and %s = %.2f" % (cols[i],cols[j],v))

# Strong correlation is observed between the following pairs
# This represents an opportunity to reduce the feature set through transformations such as PCA

""" 7. Scatter """

# Scatter plot of only the highly correlated pairs
for v,i,j in s_corr_list:
    sns.pairplot(dataset, hue="Cover_Type", size=6, x_vars=cols[i],y_vars=cols[j] )
    plt.show()

# The plots show to which class does a point belong to. The class distribution overlaps in the plots.
# Hillshade patterns give a nice ellipsoid patterns with each other
# Aspect and Hillshades attributes form a sigmoid pattern
# Horizontal and vertical distance to hydrology give an almost linear pattern.

""" 8. Data Visualization - Violin plot """
# visualize all the attributes using Violin Plot - a combination of box and density plots

# names of all the attributes
cols = dataset.columns

# number of attributes (exclude target)
size = len(cols)-1

# x-axis has target attribute to distinguish between classes
x = cols[size]

# y-axis shows values of an attribute
y = cols[0:size]

# Plot violin for all attributes
for i in range(0,size):
    sns.violinplot(data=dataset,x=x,y=y[i])
    plt.show()

# Elevation is has a separate distribution for most classes. Highly correlated with the target and hence an important attribute
# Aspect contains a couple of normal distribution for several classes
# Horizontal distance to road and hydrology have similar distribution
# Hillshade 9am and 12pm display left skew
# Hillshade 3pm is normal
# Lots of 0s in vertical distance to hydrology
# Wilderness_Area3 gives no class distinction. As values are not present, others gives some scope to distinguish
# Soil_Type, 1,5,8,9,12,14,18-22, 25-30 and 35-40 offer class distinction as values are not present for many classes

""" 9. Data Visualization - Group """
# Group one-hot encoded variables of a category into one single variable

# names of all the columns
cols = dataset.columns

# number of rows=r , number of columns=c
r,c = dataset.shape

# Create a new dataframe with r rows, one column for each encoded category, and target in the end
data = pandas.DataFrame(index=np.arange(0, r),columns=['Wilderness_Area','Soil_Type','Cover_Type'])

# Make an entry in 'data' for each r as category_id, target value
for i in range(0,r):
    w=0;
    s=0;
    # Category1 range
    for j in range(10,14):
        if (dataset.iloc[i,j] == 1):
            w=j-9  #category class
            break
    # Category2 range
    for k in range(14,54):
        if (dataset.iloc[i,k] == 1):
            s=k-13 #category class
            break
    # Make an entry in 'data' for each r as category_id, target value
    data.iloc[i]=[w,s,dataset.iloc[i,c-1]]

# Plot for Category1
sns.countplot(x="Wilderness_Area", hue="Cover_Type", data=data)
plt.show()
# Plot for Category2
plt.rc("figure", figsize=(25, 10))
sns.countplot(x="Soil_Type", hue="Cover_Type", data=data)
plt.show()

# (right-click and open the image in a new window for larger size)
# WildernessArea_4 has a lot of presence for cover_type 4. Good class distinction
# WildernessArea_3 has not much class distinction
# SoilType 1-6,10-14,17, 22-23, 29-33,35,38-40 offer lot of class distinction as counts for some are very high

""" 10. Data Cleaning """
# Remove unnecessary columns

# Removal list initialize
rem = []

# Add constant columns as they don't help in prediction process
for c in dataset.columns:
    if dataset[c].std() == 0: #standard deviation is zero
        rem.append(c)

# drop the columns
dataset.drop(rem,axis=1,inplace=True)

print(rem)

# Following columns are dropped
# ['Soil_Type7', 'Soil_Type15']


