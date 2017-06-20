
# coding: utf-8

# In[1]:

# import the pandas library
import pandas as pd


# In[2]:

from pandas import Series, DataFrame


# In[3]:

# read the csv file using pd.read_csv()


# In[4]:

titanic_df = pd.read_csv('train.csv')


# In[5]:

# peek the top rows of the dataset using pd.read_csv().head()


# In[6]:

titanic_df.head()


# In[7]:

# use pd.read_csv().info() to see the info of the dataset, e.g., number of rows, non-null or null, data type


# In[8]:

titanic_df.info()


# In[9]:

# it is important to ask relevant and insightful questions regarding the dataset:
# 1. Who were the passengers (age, class, gender...)?
# 2. What decks were the passengers and how that was related to class?
# 3. Where did they get aboard?
# 4. Who were alone and who were with family?
# 5. What factors are important for the survivors?


# In[10]:

# import the libraries that are needed for data analysis and visualization


# In[11]:

import numpy as np


# In[12]:

import matplotlib.pyplot as plt


# In[13]:

import seaborn as sns


# In[14]:

get_ipython().magic(u'matplotlib inline')


# In[15]:

sns.factorplot('Sex',data=titanic_df, kind='count')


# In[16]:

# it is seen that male passengers were almost twice of those female ones


# In[17]:

sns.factorplot('Pclass', data=titanic_df, kind ='count')


# In[18]:

# most passengers were in the 3rd class


# In[19]:

# check sub-column result with a second column specified


# In[20]:

sns.factorplot('Pclass', data=titanic_df, hue = 'Sex', kind ='count')


# In[21]:

# male and female passengers were about the same number in the 1st and 2nd classes, but male passengers were weight more than
# female in the 3rd class


# In[22]:

sns.factorplot('Sex', data = titanic_df, hue = 'Pclass', kind = 'count')


# In[23]:

# most passengers were in the 3rd class compared to the 1st and 2nd classes


# In[24]:

# common sense tells us that usually children and women have superior rights of escape ahead of men. Write a function 
# to check if how many children were on the ship.


# In[25]:

def male_female_child(passenger):
    
    age, sex = passenger
    
    if age < 16:
        return 'child'
    else:
        return sex


# In[26]:

# apply the defined function to the age and sex columns and create a new person column in the dataset
titanic_df['person']=titanic_df[['Age', 'Sex']].apply(male_female_child, axis = 1)


# In[27]:

# check the dataset to see the newly created column person


# In[28]:

titanic_df[0:20]


# In[29]:

# it should be noted that NaN values are skipped.


# In[30]:

sns.factorplot('person', data = titanic_df, kind = 'count')


# In[31]:

# count passenger numbers by sex
titanic_df['person'].value_counts()


# In[32]:

sns.factorplot('Pclass', data=titanic_df, hue = 'person', kind ='count')


# In[33]:

# create a histogram to see the age distribution
titanic_df['Age'].hist(bins=100)


# In[34]:

titanic_df['Age'].mean()


# In[35]:

# mean age of the passengers is ~30


# In[36]:

# find the maximum entry in the Age column
oldest = titanic_df['Age'].max()
oldest


# In[37]:

# create a FacetGrid with multiple plots, FacetGrid(pd.read_csv(), hue = 'column_name', aspect = aspect ratio n)
fig = sns.FacetGrid(titanic_df, hue = 'Sex', aspect = 4)
# kdeplot
fig.map(sns.kdeplot, 'Age', shade = True)
# set the figure limits
fig.set(xlim= (0, oldest))
# add legend for the figure
fig.add_legend()


# In[38]:

# create a FacetGrid with multiple plots, FacetGrid(pd.read_csv(), hue = 'column_name', aspect = aspect ratio n)
fig = sns.FacetGrid(titanic_df, hue = 'Sex', aspect = 2)
# kdeplot
fig.map(sns.kdeplot, 'Age', shade = True)
# set the figure limits
fig.set(xlim= (0, oldest))
# add legend for the figure
fig.add_legend()


# In[39]:

# do a similar KDE plot with children
fig1 = sns.FacetGrid(titanic_df, hue = 'person', aspect = 4)
# kdeplot
fig1.map(sns.kdeplot, 'Age', shade = True)
# set the figure limits
fig1.set(xlim= (0, oldest))
# add legend for the figure
fig1.add_legend()


# In[40]:

# Again, do a similar KDE plot with class
fig2 = sns.FacetGrid(titanic_df, hue = 'Pclass', aspect = 4)
# kdeplot
fig2.map(sns.kdeplot, 'Age', shade = True)
# set the figure limits
fig2.set(xlim= (0, oldest))
# add legend for the figure
fig2.add_legend()


# In[41]:

# after get a sense of the correlation between sex, age, and class, check the dataset again
titanic_df.head(40)


# In[45]:

# drop the NaN values in a particular column
deck = titanic_df['Cabin'].dropna()


# In[46]:

deck.head(10)


# In[47]:

# grab only the cabin letter in the cabin entries
levels = []

for level in deck:
    levels.append(level[0])

cabin_df = DataFrame(levels)
cabin_df


# In[48]:

# Give the column name 'Cabin'
cabin_df.columns = ['Cabin']
cabin_df



# In[49]:

# create a factorplot for the cabins with palett winter color darkened
sns.factorplot('Cabin', data = cabin_df, kind = 'count', palette = 'winter_d')


# In[50]:

# drop the 'T' column
cabin_df = cabin_df[cabin_df.Cabin != 'T']


# In[51]:

sns.factorplot('Cabin', data = cabin_df, kind = 'count', palette = 'winter_d')


# In[52]:

sns.factorplot('Cabin', data = cabin_df, kind = 'count', palette = 'winter')


# In[53]:

sns.factorplot('Cabin', data = cabin_df, kind = 'count', palette = 'summer')


# In[54]:

sns.factorplot('Cabin', data = cabin_df, kind = 'count', order = ['A','B','C','D','E','F'], palette = 'summer')


# In[55]:

titanic_df.head(10)


# In[56]:

# to quickly check where passengers of different classes are from
# What decks were the passengers and how that was related to class?
sns.factorplot('Embarked', data = titanic_df, hue = 'Pclass', kind = 'count', palette = 'winter')


# In[57]:

sns.factorplot('Embarked', data = titanic_df, hue = 'Pclass', kind = 'count')


# In[58]:

# Where did they get aboard?
sns.factorplot('Embarked', data = titanic_df, hue = 'Pclass', order = ['C', 'Q', 'S'], kind = 'count')


# In[59]:

# Who were alone and who were with family? check SibSp and Parch columns: when both column entries are 0, the passenger is alone.
# Add an alone column to the dataframe by adding the values in SibSp and Parch columns

titanic_df['Alone'] = titanic_df.SibSp + titanic_df.Parch 



# In[60]:

# check to see what it looks like
titanic_df['Alone']


# In[61]:

# filter the Alone passengers which have the 0 entries
titanic_df['Alone'].loc[titanic_df['Alone'] > 0] = 'With Family'
titanic_df['Alone'].loc[titanic_df['Alone'] == 0] = 'Alone'


# In[62]:

# the above warning is alright.
# check the updated dataframe
titanic_df


# In[64]:

# do the visualization
sns.factorplot('Alone', data = titanic_df, kind = 'count', palette = 'Blues')


# In[65]:

# the above factorplot shows that Alone people were more than those with families
# Now let's come back to the core question: What factors are important to affect the survivals?
# add a new column 'Survivor' to the dataframe
titanic_df['Survivor'] = titanic_df.Survived.map({0:'no', 1:'yes'})


# In[66]:

# check the updated dataframe
titanic_df


# In[67]:

# create a factorplot to visualize the results
sns.factorplot('Survivor', data = titanic_df, kind = 'count', palette = 'spring')


# In[68]:

# the above plot shows that more people unfortunately died compared with those who survived
# next, let's check whether survival has something to do with class
sns.factorplot('Survivor', data = titanic_df, hue = 'Pclass', kind = 'count', palette = 'summer')


# In[72]:

# it is clearly seen that third class was really unfavorable in terms of survival compared to the first and second class
# plot in another way
sns.factorplot('Pclass', 'Survived', data = titanic_df) #(x axis, y axis, data = dataframe)


# In[74]:

# it is seen that first class has the highest survival rate
# now let's check if the children-women first policy played a role
sns.factorplot('Pclass', 'Survived', hue = 'person', data = titanic_df, palette = 'autumn')


# In[75]:

# it clearly shows that children-women first policy did have a strong impact on the survival rate
# next let's check if age has an effect on survival rate using a linear plot
sns.factorplot('Age', 'Survived', data = titanic_df)


# In[77]:

sns.lmplot('Age', 'Survived', data = titanic_df)


# In[78]:

# the results indicate that the older passengers had lower chance of survival
# add one more factor which is class into consideration (similar to groupby)
sns.lmplot('Age', 'Survived', hue = 'Pclass', data = titanic_df)


# In[79]:

# the results show that:
#     still, age is a key factor affecting survival rate. Younger-age passengers had higher chance of survival. 
#     Also, class had an impact.
# refine the plot a little bit
generations = [10, 20, 40, 60, 80] # create age bins

sns.lmplot('Age', 'Survived', hue = 'Pclass', data = titanic_df, palette = 'winter', x_bins = generations)


# In[81]:

# this cleans up the upper and lower scattered data points, it is interesting to see that the standard deviation for age 80
# in the first class are pretty significant
# next, check if sex has an impact on the survival rate
sns.lmplot('Age', 'Survived', hue = 'Sex', data = titanic_df)


# In[83]:

# it is very interesting that sex shows opposite impacts on survival rates between male and female: older female had 
# greater survival rate compared to younger women
sns.lmplot('Age','Survived', hue = 'Sex', data = titanic_df, x_bins = generations)


# In[87]:

# males at 80 age group had big standard deviation 
# now let's check if the deck has an impact on the survival rate
sns.factorplot('Cabin','Survived', data = titanic_df)


# In[99]:

titanic_df_2 = titanic_df[titanic_df.Cabin.notnull()]      


# In[100]:

titanic_df_2.head(10)


# In[101]:

titanic_df_2.info()


# In[102]:

cabin_df.info()


# In[105]:

titanic_df_2['Survived']


# In[106]:

cabin_df['Cabin']


# In[90]:

# next check if having a family member on board would increase or decrease survival rate
sns.factorplot('Alone', 'Survived', data = titanic_df)


# In[92]:

# being with family does increase the chance of survival possibly due to the support from family members
# further check if effect of being with family member(s) differs by age group
sns.lmplot('Age', 'Survived', hue = 'Alone', data = titanic_df, x_bins = generations)


# In[ ]:

# it seems that age has a more significant effect on survival rate compared with the family factor

