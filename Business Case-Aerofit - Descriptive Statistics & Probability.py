#!/usr/bin/env python
# coding: utf-8

# ## Business Problem 
# The market research team at AeroFit wants to identify the characteristics of the target audience for each type of treadmill offered by the company, to provide a better recommendation of the treadmills to the new customers. The team decides to investigate whether there are differences across the product with respect to customer characteristics.
# 
# ### Dataset Information: 
# 
# Product Purchased:	KP281, KP481, or KP781
# Age:	In years
# Gender:	Male/Female
# Education:	In years
# MaritalStatus:	Single or partnered
# 
# Usage:	The average number of times the customer plans to use the treadmill each week.
# 
# Income:	Annual income (in $)
# 
# Fitness:	Self-rated fitness on a 1-to-5 scale, where 1 is the poor shape and 5 is the excellent shape.
# 
# Miles:	The average number of miles the customer expects to walk/run each week
# 
# 1. Perform descriptive analytics to create a customer profile for each AeroFit treadmill product by developing appropriate tables and charts.
# 
# 2. For each AeroFit treadmill product, construct two-way contingency tables and compute all conditional and marginal probabilities along with their insights/impact on the business.
# 

# In[3]:


import numpy as np
import pandas as pd
import scipy 
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


data=pd.read_csv(r"C:\Users\Tnluser\Documents\Datasets\Python\aerofit_treadmill.csv")
data.head()


# In[6]:


data.size


# In[7]:


data.shape


# In[9]:


data.isnull().sum()


# In[16]:


data.info()


# In[21]:


#statiscal information about  numerical columns
data.describe()


# In[23]:


data.Usage.unique()


# In[24]:


data.Usage.value_counts()


# In[25]:


data.Product.value_counts()


# In[84]:


data.Age.unique()


# In[102]:


def bucket(x):
    if x<=25:
        return '18-30'
    elif x<=40:
        return '30-40'
    else:
        return '40-50'


data['age_bucket']=data['Age'].apply(bucket)
data.head()


# ### Detect Outliers (using boxplot, “describe” method by checking the difference between mean and median)

# In[104]:


#statiscal information about  numerical columns
data.describe().loc[['mean','50%']]


# Outcome: Here miles and income there is significant difference between mean and median hence there are outlier

# In[114]:


cols=['Income', 'Miles']
l=len(cols)
plt.figure(figsize=(15,5))
count=1
for i in cols:
    plt.subplot(1,l,count)
    sns.boxplot(data=data,x='Product',y=i)
    count +=1
    


# In[266]:


sns.histplot(data.Age,bins=15)
## Potential buyers' age bucket is 20 to 35 years


# In[288]:


plt.figure(figsize=(15,10))
plt.suptitle("User base segmentation based on Age and Marital Status",fontsize=20)
plt.subplot(3,3,1)
plt.title("Age Chart")
sns.histplot(data.Age,bins=15)
plt.subplot(3,3,3)
plt.title("Marital Status Chart")
sns.histplot(data.MaritalStatus,bins=15)

##features like marital status, age have any effect on the product purchased
count=4
temp=data[(data.Age>=20) & (data.Age<=35)]
plt.figure(figsize=(15,10))
for i in data.Product.unique():
    plt.subplot(3,3,count)
    plt.title(i)
    sns.histplot(temp[temp.Product==i].Age,bins=3,palette='pastel',color='pink')
    count +=1

    
##features like marital status, age have any effect on the product purchased

count=7
plt.figure(figsize=(12,8))
for i in data.Product.unique():
    plt.subplot(3,3,count)
    plt.title(i)
    sns.histplot(data[data.Product==i].MaritalStatus,color='green')
    count +=1


# ### Insights: 
# 
# Potential buyers' age bucket is 20 to 35 years
# 
# KP281, age group : 20 to 30
# 
# KP481 age group:  30 to 35
# 
# KP781 age group: 22 to 26

# In[279]:


data.MaritalStatus.value_counts(normalize=True)


# In[115]:


## income and miles both are numerical 
## when both numerical or continous variables, to understand distribution of data lets use scatter plot
## line plot or bar plot can be used here
sns.scatterplot(data=data, x='Miles', y='Income', hue='Product')


# ### insights:
# high miles requirement customes prefer Kp281 , aslo their income is high

# In[120]:


market_share=data['Product'].value_counts(normalize=True)*100
market_share


# In[36]:


data['Gender'].value_counts(normalize=True)*100


# In[37]:


pd.crosstab(index=data.Gender,columns=data.Product)


# In[38]:


pd.crosstab(index=data.Gender, columns=data.Product,margins=True)


# In[45]:


## what is the percentage of Females who are buying KP281 
## Answer: 22 percent of the total population


# In[150]:


data_crosstab=pd.crosstab(index=data.Gender,columns=data.Product, normalize=True).apply(lambda x: round(x,2)*100)
data_crosstab


# In[146]:


data_crosstab.columns.name


# In[152]:


product_gender=data_crosstab.stack(level=-1).reset_index()
product_gender


# In[165]:


product_gender.rename(columns={0:'Gender_perc'},inplace=True)
product_gender


# In[154]:


data_crosstab=pd.crosstab(index=data.Miles,columns=data.Product, normalize=True).apply(lambda x: round(x,2)*100)
data_crosstab.head()


# In[166]:


product_miles=data_crosstab.stack(level=-1).reset_index()
product_miles.rename(columns={0:'miles_perc'},inplace=True)
product_miles.head()


# In[167]:


data_crosstab=pd.crosstab(index=data.age_bucket,columns=data.Product, normalize=True).apply(lambda x: round(x,2)*100)
data_crosstab.head()


# In[168]:


product_age=data_crosstab.stack(level=-1).reset_index()
product_age.rename(columns={0:'age_perc'},inplace=True)
product_age.head()


# In[170]:


data_crosstab=pd.crosstab(index=data.MaritalStatus,columns=data.Product, normalize=True).apply(lambda x: round(x,2)*100)
product_marital=data_crosstab.stack(level=-1).reset_index()
product_marital.rename(columns={0:'marital_perc'},inplace=True)
product_marital.head()


# In[173]:


data_crosstab=pd.crosstab(index=data.Fitness,columns=data.Product, normalize=True).apply(lambda x: round(x,2)*100)
product_fitness=data_crosstab.stack(level=-1).reset_index()
product_fitness.rename(columns={0:'fitness_perc'},inplace=True)
product_fitness.head()


# In[174]:


data_crosstab=pd.crosstab(index=data.Usage,columns=data.Product, normalize=True).apply(lambda x: round(x,2)*100)
product_usage=data_crosstab.stack(level=-1).reset_index()
product_usage.rename(columns={0:'usage_perc'},inplace=True)
product_usage.head()


# In[193]:


df_market_share=data['Product'].value_counts(normalize=True)*100
df_market_share=pd.DataFrame(df_market_share).reset_index()
df_market_share.rename(columns={'index':'Product','Product':'market_share'},inplace=True)
df_market_share['market_share']=df_market_share['market_share'].apply(lambda x: round(x,2))
df_market_share


# In[240]:


df_gender=data['Gender'].value_counts(normalize=True)*100
df_gender=pd.DataFrame(df_gender).reset_index()
df_gender.rename(columns={'index':'Gender','Gender':'gender_perc'},inplace=True)
df_gender['gender_perc']=df_gender['gender_perc'].apply(lambda x: round(x,2))
df_gender


# In[51]:


## Reusability of function format

def distribution(col):
    print(data[col].value_counts(normalize=True)*100)
    print()
    
cols=['Product','Gender','Usage']
for i in cols:
    distribution(i)


# ### Dashboard: Customer Profile for tredmill products

# In[242]:



# product_gender,product_usage,product_fitness,product_marital,product_age,product_miles
plt.figure(figsize=(20,14))
plt.suptitle('Dashboard: Customer Profiling - Categorization of users',fontsize=30)
plt.subplot(3,3,1)
sns.barplot(data=product_gender,x='Product',y='Gender_perc',hue='Gender')
plt.subplot(3,3,2)
sns.barplot(data=product_age,x='Product',y='age_perc',hue='age_bucket')
plt.subplot(3,3,3)
sns.barplot(data=product_marital,x='Product',y='marital_perc',hue='MaritalStatus')
plt.subplot(3,3,4)
sns.barplot(data=product_usage,x='Product',y='usage_perc',hue='Usage')
plt.subplot(3,3,5)
sns.scatterplot(data=data, x='Miles', y='Income', hue='Product')
plt.subplot(3,3,6)
sns.boxplot(data=data,x='Product',y='Income')
plt.subplot(3,3,7)
sns.heatmap(data.corr(),annot=True)
plt.subplot(3,3,8)
sns.barplot(data=df_market_share,x='Product',y='market_share')
plt.subplot(3,3,9)
sns.barplot(data=df_gender,x='Gender',y='gender_perc')


# ### Insights
# 
# 1. Correlation
#     
# High positive correlation b/w parameters found as following:
# age & income.
# 
# eduction & income.
# 
# usage & fitness, usage & miles
# 
# fitness & miles
# 
# 2. Market share of product is as following:
# 
# KP281    44.44
# 
# KP481    33.33
# 
# KP781    22.22
# 
# 3. Age Impact
# 
# Potential buyers' age bucket is 20 to 35 years
# 
# KP281, age group : 20 to 30
# 
# KP481 age group: 30 to 35
# 
# KP781 age group: 22 to 26
# 
# 4. Gender Segmentation
# 
# Male      57.78%
# 
# Female    42.22%
# 
# 
# KP281 there is no gender baising, 
# 
# for KP781 male user is significantly higher than female user base
# 
# 5. What is the probability of a male customer buying a KP781 treadmill?
# 
# Male	KP781	18.0% of total population
# 
# i.e Probabilty is 0.18
# 
# 6. Marital Status-User base percentage
# 
# Partnered    59.44%
# 
# Single       40.55%
# 
# 
# 7. ncome Impact</h3>
# 
# High income user base prefer KP781 Product
# 
# There is no significant difference in the mean income of the user base for the other two products.
# 
# 8. Segmentation based on Usage parameter
# 
# Usage: The average number of times the customer plans to use the treadmill each week.
# 
# KP281 and KP481 the maximum user base prefer usage=3
# 
# KP871 user base prefer usage=4

# ## Recommendations

# Based on the insights, here are some recommendations for AeroFit's business problem:
# 
# 1. Target Audience Segmentation:
# 
# Segment the target audience based on age, gender, marital status, and income.
# 
# For KP281, target individuals aged 20 to 30, both males and females, with a preference for those who are single or partnered, and with varying income levels.
# 
# For KP481, target individuals aged 30 to 35, both males and females, with a preference for those who are single or partnered, and with varying income levels.
# 
# For KP781, target individuals aged 22 to 26, predominantly males, preferably with higher income levels.
# 
# 2. Product Features and Marketing:
# 
# Based on positive correlation between age and income, emphasize the affordability and suitability of each treadmill model.
# 
# Emphasize the correlation between education and income to appeal to educated individuals who value fitness.
# 
# Market KP281 and KP481 as suitable for moderate users (usage = 3 times a week) while positioning KP781 as ideal for more frequent users (usage = 4 times a week).
# 
# Utilize the higher market share of KP281 to leverage brand recognition and introduce the other models as complementary options.
# 
# 3. Gender Targeting:
# 
# Focus on neutral marketing for KP281 but highlight the gender-neutral aspects of the product.
# 
# Tailor marketing strategies for KP781 to target the predominantly male user base, emphasizing features that appeal to this demographic, such as advanced technology or performance optimization.
# 
# 4. Income-based Strategies:
# 
# Offer financing or installment plans for KP781 to attract high-income individuals who prefer this product.
# 
# Implement promotional offers or discounts targeting specific income brackets to encourage purchase across all product lines.
# 
# 5. Market Expansion:
# 
# Explore opportunities to expand market share by targeting new demographics or geographical regions based on the identified characteristics of each treadmill's target audience.
# 
# Consider partnerships with gyms, fitness centers, or influencers aligned with the target demographics to increase brand visibility and reach.
# 
# By implementing these recommendations, AeroFit can effectively tailor its marketing efforts, product positioning, and customer engagement strategies to better cater to the distinct characteristics and preferences of each treadmill's target audience.

# In[ ]:




