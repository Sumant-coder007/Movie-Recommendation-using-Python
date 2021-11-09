#%%
print("hello")
# %%
from os import altsep
from typing import final
import numpy as np
from numpy.core.arrayprint import format_float_positional
import pandas as pd 
import warnings 
warnings.filterwarnings('ignore')
# %%
column_names=['user_id','item_id','rating','timestamp']
df=pd.read_csv(r"C:\Users\my pc\projects\Projects_ML_CB\ml-100k\u.data",sep='\t',names=column_names)
df.head()
# %%
unique_users=set(list(df['user_id']))
print(len(unique_users))
df['user_id'].nunique()
#%%
unique_users=set(list(df['item_id']))
print(len(unique_users))
df['item_id'].nunique()
# %%
mean_rating=np.mean(df['rating'])
print(mean_rating)
# %%
movies=pd.read_csv(r"C:\Users\my pc\projects\Projects_ML_CB\ml-100k\u.item",sep='\|',header=None, encoding = "ISO-8859-1")
movies.head()
# %%
movies=movies[[0,1]]
movies.columns=['item_id','Title']
print(movies.shape)
movies.head()  
#%%
final_df=pd.merge(df,movies,on='item_id')
final_df.tail()
# %%
'''exploratory data analysis'''
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('white')

#%%
movie_wise_rating=pd.DataFrame(final_df.groupby('Title').mean()['rating'])
movie_wise_rating.head()
# %%
# movie_wise_rating=movie_wise_rating.sort_values(ascending=False)
# movie_wise_rating.head()
#%%
Number_of_ratings=final_df.groupby('Title').count()['rating']
#final_df.groupby('Title').count()['rating'].sort_values(ascending=False)
# %%
movie_wise_rating['Number_of_ratings']=pd.DataFrame(Number_of_ratings)
#%%
movie_wise_rating.sort_values(by=['rating'],ascending=False)
# %%
plt.figure(figsize=(5,5))
plt.hist(movie_wise_rating['rating'],bins=50)
plt.xlabel("No. of ratings")
plt.ylabel('Number of movies')
plt.title("Ratings vs Movies")
plt.show()
# %%
plt.figure(figsize=(5,5))
plt.hist(movie_wise_rating['rating'],bins=70)
plt.xlabel("Rating")
plt.ylabel('Number of persons rated')
plt.title("Rating vs Movies")
plt.show()
# %%
sns.set_style('dark')
sns.jointplot(x='rating',
           y='Number_of_ratings',
           data=movie_wise_rating,
           kind='hex',
           alpha=0.5 )
# %%
movies_pivot=final_df.pivot_table(index="user_id",columns='Title',values='rating')
# %%
movies_pivot
# %%
data=(movie_wise_rating.sort_values('Number_of_ratings',ascending=False))
print(data)
# %%
top_rated_movie=movies_pivot["Star Wars (1977)"]
top_rated_movie
# %%
similar_to_toprated=pd.DataFrame(movies_pivot.corrwith(top_rated_movie),columns=['Correlation'])
# %%
similar_to_toprated.dropna(inplace=True)
similar_to_toprated
 # %%
corr_star_wars=similar_to_toprated.join(movie_wise_rating['Number_of_ratings'])
# %%
corr_star_wars[corr_star_wars['Number_of_ratings']>=100].sort_values('Correlation',ascending=False)[0]

# %%
def predict_movies(movie_name):
    top_rated_movie=movies_pivot[movie_name]
    similar_to_toprated=pd.DataFrame(movies_pivot.corrwith(top_rated_movie),columns=['Correlation'])
    similar_to_toprated.dropna(inplace=True)
    corr_top_rated=similar_to_toprated.join(movie_wise_rating['Number_of_ratings'])
    recommendations=corr_top_rated[corr_top_rated['Number_of_ratings']>=100].sort_values('Correlation',ascending=False)
    return recommendations
predictions=np.array(predict_movies("Titanic (1997)"))
predictions.head()
# %%
