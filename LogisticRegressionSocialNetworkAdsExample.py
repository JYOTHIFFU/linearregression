
# coding: utf-8

# In[1]:


#Use-case : Create a model that can predict whether the customer will purchase the product
#from my e-store based on his/her age and estimated salary

#Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Read the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')


# In[3]:


dataset.info()


# In[5]:


dataset.head(5)


# In[15]:


#Split the data with features and label
features = dataset.iloc[:,[2,3]].values
label = dataset.iloc[:,4].values


# In[16]:


#Since I am going to consider age and salary as the feature, it is important for us to
#understand whether feature scaling is needed or not. To take a judgement we can check
#the minimum and maximun of individual feature fields. If the magnitude of the respective
#fields is huge, we shall go for feature scaling!!!
print(dataset['Age'].min())
print(dataset['Age'].max())
print(dataset['EstimatedSalary'].min())
print(dataset['EstimatedSalary'].max())


# In[17]:


#Create train/test split
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test= train_test_split(features,
                                                label,
                                                test_size=0.25,
                                                random_state=0)


# In[18]:


#The best method considered for Feature Scaling is StandardScaler.
from sklearn.preprocessing import StandardScaler
sc  = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[19]:


#Lets create the model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)


# In[20]:


#Check the score
model.score(X_train,y_train)


# In[21]:


model.score(X_test,y_test)


# In[22]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,model.predict(X_test))
cm


# In[27]:


63+25 / (63+5+7+25) *100


# In[24]:


5+7


# In[31]:


#Age and Purchase
plt.scatter(X_test[:,0],y_test,color='b')
plt.plot(X_test[:,0],model.predict(X_test),color='r')


# In[32]:


#Sal and Purchase
plt.scatter(X_test[:,1],y_test,color='b')
plt.plot(X_test[:,1],model.predict(X_test),color='r')

