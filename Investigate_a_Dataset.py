#!/usr/bin/env python
# coding: utf-8

# # Project: TMDb movie data analysis
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# To complete my Data Analysis project I am using TMDb movies dataset.
# 
# This data set contains information about 10 thousand movies collected from The Movie Database (TMDb), including user ratings and revenue. It consist of 21 columns such as imdb_id, revenue, budget, vote_count etc.
# 
# 
# #### Questions that can analyised from this data set
# 
# 1. Most and least popular movies
# 2. Movies with most and least budget
# 3. Movies with most and least revenue
# 4. Runtime of all Movies
# 5. Popularity of Movies each year
# 6. Most popular genres by century
# 7. Most frequent actors by century
# 8. Average budget of movies by century
# 9. Average revenue of movies by century

# In[1]:


# Use this cell to set up import statements for all of the packages that you
#   plan to use.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
# Remember to include a 'magic word' so that your visualizations are plotted
#   inline with the notebook. See this page for more:
#   http://ipython.readthedocs.io/en/stable/interactive/magics.html


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# 
# ### General Properties

# In[7]:


#Reading the database
df = pd.read_csv('tmdb-movies.csv')
df.head()


# In[3]:


#Checking the shape, mostly to see how many columns we have
df.shape


# In[8]:


#Checking a summary of the database
df.describe()


# In[10]:


#Checking more info to see if values are missing in some rows and the data types in case a change is needed
# Also, we do a histogram of the values to notice visually the spread of the data
df.info()
df.hist(figsize=(50,20))


# In[11]:


#Dropping the columns that are not needed and saving the changes
df.drop(['imdb_id', 'homepage', 'tagline', 'keywords', 'overview', 'budget_adj', 'revenue_adj'], 1, inplace=True)
df.head()


# ### Data Cleaning (Removing duplicates, missing values, dropping unnessesary columns)

# In[22]:


# After discussing the structure of the data and any problems that need to be
#   cleaned, perform those cleaning steps in the second part of this section.
# Here we remove any duplicates that exist. 
df.drop_duplicates(keep ='first', inplace=True)
df.shape


# Revenue, budget and runtime are some really important factors of our future analysis and we noticed from the graphs that are a lot of values that are 0. We are going to remove these movies from our list and to do so, we are going to replace zero values with NaN and then drop the rows with NaN budget and revenue

# In[23]:


temp_list=['budget', 'revenue', 'runtime']

#this will replace all the value from '0' to NAN in the list
df[temp_list] = df[temp_list].replace(0, np.NAN)

#Removing all the row which has NaN value in temp_list 
df.dropna(subset = temp_list, inplace = True)

rows, col = df.shape
print('So after removing such entries, we now have only {} no.of movies.'.format(rows-1))


# Now we are going to convert the release dates now to the format that we can handle better

# In[12]:


#Coverting the release date in a format that suits us
df.release_date = pd.to_datetime(df['release_date'])
df.head(5)


# Now we are going to check the data types and see if we need to change any in order to help us in the analysis

# In[13]:


df.dtypes


# In[14]:


#We are going to change the data types of the budget and the revenue to intagers instead of floats
change_type=['budget', 'revenue']
#changing data type
df[change_type]=df[change_type].applymap(np.int64)
#printing the changed information
df.dtypes
df.head(5)


# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# 
# ### Research Question 1: Most and least popular movies

# In[15]:


#Sorting the movies by popularity and storing them separately
popular_movies = df.sort_values('popularity', ascending = False)
popular_movies.head()


# In[16]:


import pprint
#defining the function that will give us the min and max in our database
def calculate(column):
    #for highest 
    high= df[column].idxmax()
    high_details=pd.DataFrame(df.loc[high])
    
    #for lowest 
    low= df[column].idxmin()
    low_details=pd.DataFrame(df.loc[low])
    
    #collectin data in one place
    info=pd.concat([high_details, low_details], axis=1)
    
    return info

#calling the function
calculate('popularity')


# ### Research Question 2: Movies with most and least budget

# In[17]:


# Here we use the function above to see the movies with the biggest and lowest budget
calculate('budget')


# ### Research Question 3: Movies with most and least revenue

# In[18]:


# Here we use the same function to see the movies with the biggest and lowest budget
calculate('revenue')


# ### Research question 4: Runtime of all Movies

# In[31]:


#Checking the runtime of movies 
plt.figure(figsize=(9,5), dpi = 100)

#On x-axis 
plt.xlabel('Runtime of Movies', fontsize = 14)
#On y-axis 
plt.ylabel('No. of Movies in the Dataset', fontsize=14)
#Name of the graph
plt.title('Runtime of all movies', fontsize=14)

#giving a histogram plot
plt.hist(df['runtime'], rwidth = 1, bins =35)
#displays the plot
plt.show()


# We notice that the majority of the movies last around 100 minutes and there are a few edge cases with movies lasting less than 50 minutes and some more than 200

# ### Research question 5: Popularity of Movies each year

# In[20]:


#Checking the popularity of movies by year
popula = df.groupby('release_year')['popularity'].sum()

#figure size(width, height)
plt.figure(figsize=(12,6), dpi = 130)

#on x-axis
plt.xlabel('Release Year of Movies in the data set', fontsize = 12)
#on y-axis
plt.ylabel('Popularity of movies by year', fontsize = 12)
#title of the line plot
plt.title('Representing Total Popularity of movies Vs Year of their release.')

#plotting the graph
plt.plot(popula)

#displaying the line plot
plt.show()


# We notice that the popularity of the movies start increasing after 1990 significantly and even more after 2000. 
# 
# That led me to the idea to split the database into movies of the 20th and 21st century

# We are going to split the movies in movies that released before the 21st century and during the 21st century

# In[21]:


before_data = df[df['release_year'] < 2000]

after_data = df[df['release_year'] >= 2000]
#reindexing new data
before_data.index = range(len(before_data))

after_data.index = range(len(after_data))

#we will start from 1 instead of 0
before_data.index = before_data.index + 1

after_data.index = after_data.index + 1


# In[22]:


#we are going to create a class that separates a column by '|' as we will use it for the genders
def data(column, data_frame):
    #will take a column, and separate the string by '|'
    data = data_frame[column].str.cat(sep = '|')
    
    #giving pandas series and storing the values separately
    data = pd.Series(data.split('|'))
    
    #arranging in descending order
    count = data.value_counts(ascending = False)
    
    return count


# ### Research question 6: Most popular genres by century

# In[23]:


count_after = data('genres', after_data)
count_after.head()


# In[24]:


count_before = data('genres', before_data)
count_before.head()


# In[25]:


count_after.sort_values(ascending = False, inplace = True)

#ploting
lt = count_after.head(5).plot.barh(color = '#174EA6', fontsize = 13)

#title
lt.set(title = 'Frequent Used Genres in Movies in the 21st century')

# on x axis
lt.set_xlabel('Nos.of Movies in the dataset', color = 'black', fontsize = '13')

#figure size(width, height)
lt.figure.set_size_inches(10, 5)

#ploting the graph
plt.show()


# Drama, Comedy, Thriler, Action and Adventure are the most popular genres in the 21st century

# In[26]:


count_before.sort_values(ascending = False, inplace = True)

#ploting
lt = count_before.head(5).plot.barh(color = '#A50E0E', fontsize = 13)

#title
lt.set(title = 'Frequent Used Genres in Movies in the 20th Century')

# on x axis
lt.set_xlabel('Nos.of Movies in the dataset', color = 'black', fontsize = '13')

#figure size(width, height)
lt.figure.set_size_inches(10, 5)

#ploting the graph
plt.show()


# Drama, Comedy, Thriler, Action and Romance are the most popular genres in the 20th century. We notice that the first 4 are the same like the 21st century

# ### Research question 7: Most frequent actors by century

# In[27]:


count_after = data('cast', after_data)
count_after.head(10)


# In[28]:


count_before = data('cast', before_data)
count_before.head(10)


# ### Research question 8: Average budget of movies by century

# In[29]:


before_data['budget'].mean()


# In[30]:


after_data['budget'].mean()


# ### Research question 9: Average revenue of movies by century

# In[31]:


before_data['revenue'].mean()


# In[32]:


after_data['revenue'].mean()


# <a id='conclusions'></a>
# ## Conclusions
# 
# In this dataset, we investigated some general attributes of our dataset and then splitted the dataset by century to further analyse it.
# 
# We were able to identify the following facts:
# 
# 1. The popularity of the movies in the 21st cetury is significantly higher than the one in the 20th century
# 2. The most frequent genres remained the same in both centuries
# 3. The only actor that is in the top 10 in appearences in both centuries is Robert De Niro
# 4. The average budget of the movies in the 21st century almost doubled compared to the 20th century
# 5. The average revenue of the movies in the 21st century increased by almost one third compared to the 20th century
# 
# Limitations: A large number of the movies was removed due to incomplete data. It is quite possible that if we had the complete data of the movies, the conclusions would be different.
# In addition to that, we are not sure if the data given included all the movies until today.

# In[33]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])


# In[ ]:




