#!/usr/bin/env python
# coding: utf-8

# In[287]:


#####################################################################################################
######################### NETFLIX REVIEWS DATA SET  #################################################
#####################################################################################################


# In[288]:


#################################################################
############ Part I - Importing
#################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[289]:


import nltk


# In[290]:


df = pd.read_csv('netflix_reviews.csv')


# In[291]:


df.head()


# In[292]:


df.info()


# In[293]:


#####################################################################
########################### Part II - Duplicates
#####################################################################


# In[294]:


df[df.duplicated()]


# In[295]:


df = df.drop_duplicates()


# In[296]:


df[df.duplicated()]              #### no duplicates left


# In[297]:


####################################################################
############## Part III - Missing Values
####################################################################


# In[298]:


from matplotlib.colors import LinearSegmentedColormap

Amelia = LinearSegmentedColormap.from_list('black_yellow', ['black', 'yellow'])


# In[299]:


fig, ax = plt.subplots(figsize=(20,8))

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap=Amelia,ax=ax)

ax.set_xlabel('Columns')
ax.set_ylabel('Rows')
ax.set_title('Missing Data Heatmap')

#### why Amelia, if you coming from R then you might have used Amelia package which detects the missing value 
#### On July 2, 1937, Amelia disappeared over the Pacific Ocean while attempting to become the first female pilot to circumnavigate the world


# In[300]:


df.isnull().any()


# In[301]:


df.columns


# In[302]:


df = df[['content','score']]

df.head()                          #### these are the ones we will be looking at


# In[303]:


df.isnull().any()


# In[304]:


df.dropna(subset=['content'],inplace=True)


# In[305]:


df.isnull().any()                #### no more null values


# In[306]:


fig, ax = plt.subplots(figsize=(20,8))

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap=Amelia,ax=ax)

ax.set_xlabel('Columns')
ax.set_ylabel('Rows')
ax.set_title('Missing Data Heatmap')


# In[307]:


####################################################################
############## Part IV - Feature Engineering
####################################################################


# In[308]:


df['scores'] = df.score.apply(lambda x:0 if x in [1,2] else (1 if x==3 else 2))


# In[309]:


df.head(10)


# In[310]:


df['length'] = df.content.apply(len)


# In[311]:


df.head(10)


# In[312]:


df.length.max()


# In[313]:


df.length.describe()


# In[314]:


df.score.value_counts()


# In[315]:


df.scores.value_counts()                             #### 0 is poor, 1 is neutral, 2 is good


# In[316]:


######################################################################
############## Part V - EDA
######################################################################


# In[317]:


heat = df.groupby(['length'])['scores'].sum().sort_values(ascending=False).head(20)             #### these are the top 20 most heated length in reviews according to the reviews

heat


# In[318]:


heat.plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red',color='black',markersize=15,linestyle='dashed',linewidth=4)

plt.title('Netflix Heat Graph')

plt.xlabel('Review Length')

plt.ylabel('Density')

#### it depicts the length of reviews according to the score values


# In[319]:


df['length'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red',color='black')

plt.title('Netflix Length Graph')

plt.xlabel('Number of people')

plt.ylabel('Density')

#### seems like most of the reviews are around 500 densisty


# In[320]:


df.length.mean()


# In[321]:


df.length.std()                     #### interesting


# In[322]:


df.length.quantile(0.99)            #### this is what we seeing, amazing


# In[323]:


df[df.length <= 500]


# In[324]:


df.info()


# In[325]:


custom = {0:'red',
          1:'black',
          2:'green'}

plt.figure(figsize=(17,5))
sns.histplot(x='length',data=df,hue='scores',palette=custom,multiple='dodge',bins=5)

#### it seems like the majority of people who leave best or poor reviews end up writing around 250 lengh of words
#### but then we have outliers who are writing 1750 words, lets see what they have to say


# In[326]:


df[df.length == df.length.max()]['content'].iloc[0]                          #### quite interesting but obviously an outlier


# In[327]:


custom = {0:'red',
          1:'black',
          2:'green'}

g = sns.jointplot(x=df.length,y=df.score,data=df,hue='scores',palette=custom)

g.fig.set_size_inches(17,9)


# In[328]:


g = sns.jointplot(x='length',y='scores',data=df,kind='kde',fill=True)

g.fig.set_size_inches(17,9)

#### we can clearly see that the difference between 0 and 2 score is very tight


# In[329]:


sns.catplot(x='scores',y='length',data=df,kind='box',height=7,aspect=2,legend=True)

#### this makes it more clear


# In[330]:


g = sns.jointplot(x='length',y='scores',data=df,kind='reg',x_bins=[range(1,750)],color='black',joint_kws={'line_kws':{'color':'red'}})

g.fig.set_size_inches(17,9)

g.ax_joint.set_xlim(0,750)
g.ax_joint.set_ylim(0,df.scores.max())

#### this is quite interesting, length has a correlation to people leaving neutral reviews


# In[332]:


from scipy.stats import pearsonr


# In[333]:


co_eff, p_value = pearsonr(df.length,df.scores)

co_eff                                   #### not looking good


# In[334]:


p_value                                  #### but from this we can see some correlation, we accept alternative hypothesis


# In[335]:


custom = {0:'red',
          1:'black',
          2:'green'}

pl = sns.FacetGrid(df,hue='scores',aspect=4,height=4,palette=custom)

pl.map(sns.kdeplot,'length',fill=True)

pl.set(xlim=(0,750))

pl.add_legend()

#### the density of good reviews length is majority within 0-100


# In[336]:


g = sns.lmplot(x='length',y='scores',data=df,x_bins=[range(0,750)],height=7,aspect=2,line_kws={'color':'red'},scatter_kws={'color':'black'})

g.set(xlim=(0,750), ylim=(0,df['scores'].max()))


# In[337]:


######################################################################
############## Part VI - Feature Engineering II
######################################################################


# In[338]:


nltk.download('punkt')


# In[339]:


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# In[340]:


#### function for doing everthing like preprocess and stop words

def preprocess_text(text):
    if not isinstance(text,str):
        return ''
    # Tokenize text
    tokens = word_tokenize(text)
    # Convert to lowercase
    tokens = [word.lower() for word in tokens]
    # Remove punctuation
    tokens = [word for word in tokens if word.isalnum()]
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)


# In[341]:


df['clean_review'] = df['content'].apply(preprocess_text)


# In[342]:


df.head()


# In[343]:


df['content'][1:5]


# In[344]:


#### before we implement inside the model, lets make a nice wordcloud

text_corpus = ' '.join(df['clean_review'])


# In[345]:


from wordcloud import WordCloud


# In[346]:


wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_corpus)


# In[347]:


plt.figure(figsize=(20, 9))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Remove axis
plt.show()


# In[145]:


######################################################################
############## Part VII - Model - Classification
######################################################################


# In[171]:


new_df = df[df.length <=750]

new_df.head()


# In[172]:


X = df.clean_review

X.head()


# In[174]:


X.isnull().any()


# In[175]:


y = df.scores

y.head()


# In[181]:


from sklearn.model_selection import train_test_split


# In[176]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[177]:


from sklearn.pipeline import Pipeline


# In[152]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[178]:


model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])


# In[179]:


model.fit(X_train, y_train)                      #### on clean review


# In[180]:


y_predict = model.predict(X_test)


# In[182]:


from sklearn import metrics


# In[183]:


metrics.accuracy_score(y_test,y_predict)


# In[184]:


print(metrics.classification_report(y_test,y_predict))                             #### this was expected given how imbalanced the ratio is


# In[185]:


X = new_df.content

X.head()


# In[186]:


y = new_df.scores

y.head()


# In[194]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[195]:


model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', MultinomialNB())
])


# In[196]:


model.fit(X_train, y_train)                            #### on raw review


# In[197]:


y_predict = model.predict(X_test)


# In[198]:


metrics.accuracy_score(y_test,y_predict)


# In[199]:


print(metrics.classification_report(y_test,y_predict))                     #### imbalance is throwinf off our model for neutral reviews which is understandable


# In[200]:


from sklearn.ensemble import RandomForestClassifier


# In[211]:


model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', RandomForestClassifier(random_state=42,n_jobs=-1))
])


# In[212]:


model.fit(X_train, y_train)


# In[213]:


y_predict = model.predict(X_test)


# In[214]:


metrics.accuracy_score(y_test,y_predict)


# In[215]:


print(metrics.classification_report(y_test,y_predict))                             #### some improvement but not ideal


# In[216]:


cm = metrics.confusion_matrix(y_test,y_predict)

labels = ['Bad','OK','Good']

disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)

fig, ax = plt.subplots(figsize=(25,12))

disp.plot(ax=ax)


# In[217]:


y_test.value_counts()


# In[218]:


from sklearn.ensemble import GradientBoostingClassifier


# In[220]:


model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=42))
])


# In[221]:


model.fit(X_train, y_train)


# In[222]:


y_predict = model.predict(X_test)


# In[223]:


metrics.accuracy_score(y_test,y_predict)


# In[224]:


print(metrics.classification_report(y_test,y_predict))                             #### again the same problem


# In[225]:


from imblearn.over_sampling import SMOTE


# In[226]:


from imblearn.pipeline import Pipeline as ImbPipeline


# In[230]:


model = ImbPipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('smote', SMOTE(random_state=42)),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42,n_jobs=-1,verbose=2))
])


# In[231]:


model.fit(X_train, y_train)


# In[232]:


y_predict = model.predict(X_test)


# In[233]:


print(metrics.classification_report(y_test,y_predict))               #### smote is bring it back somehow


# In[234]:


from sklearn.model_selection import GridSearchCV


# In[235]:


param_grid = {
    'clf__n_estimators': [100, 200, 300],
    'clf__max_depth': [None, 10, 20, 30],
    'clf__min_samples_split': [2, 5, 10],
    'clf__min_samples_leaf': [1, 2, 4],
}


# In[236]:


grid_search = GridSearchCV(model, param_grid, cv=3, scoring='f1_macro')


# In[237]:


grid_search.fit(X_train, y_train)


# In[238]:


best_model = grid_search.best_estimator_


# In[239]:


y_predict = best_model.predict(X_test)


# In[240]:


print(metrics.classification_report(y_test,y_predict))                     #### took way long but gave us some improvement


# In[242]:


get_ipython().run_cell_magic('time', '', "\nmodel = ImbPipeline([\n    ('tfidf', TfidfVectorizer(stop_words='english')),\n    ('smote', SMOTE(random_state=42)),\n    ('clf', RandomForestClassifier(n_estimators=100, random_state=42,n_jobs=-1,class_weight='balanced'))\n])")


# In[243]:


get_ipython().run_cell_magic('time', '', '\nmodel.fit(X_train, y_train)')


# In[244]:


y_predict = model.predict(X_test)


# In[245]:


print(metrics.classification_report(y_test,y_predict))                     #### random forest balanced weight outcome


# In[246]:


from xgboost import XGBClassifier


# In[251]:


model = ImbPipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('smote', SMOTE(random_state=42)),
    ('clf', XGBClassifier(scale_pos_weight=y_train.value_counts().max()/y_train.value_counts().min(),
                          use_label_encoder=False, eval_metric='mlogloss', random_state=42))
])


# In[252]:


get_ipython().run_cell_magic('time', '', '\nmodel.fit(X_train, y_train)')


# In[253]:


y_predict = model.predict(X_test)


# In[254]:


print(metrics.classification_report(y_test,y_predict))


# In[255]:


from lightgbm import LGBMClassifier


# In[256]:


model = ImbPipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('smote', SMOTE(random_state=42)),
    ('clf', LGBMClassifier(class_weight='balanced', random_state=42))

])


# In[257]:


get_ipython().run_cell_magic('time', '', '\nmodel.fit(X_train, y_train)')


# In[258]:


y_predict = model.predict(X_test)


# In[259]:


print(metrics.classification_report(y_test,y_predict))                 #### best one yet


# In[260]:


from imblearn.combine import SMOTEENN


# In[261]:


model = ImbPipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('smoteenn', SMOTEENN(random_state=42)),
    ('clf', LGBMClassifier(class_weight='balanced', random_state=42))

])


# In[262]:


get_ipython().run_cell_magic('time', '', '\nmodel.fit(X_train, y_train)')


# In[263]:


y_predict = model.predict(X_test)


# In[264]:


print(metrics.classification_report(y_test,y_predict))                      #### not ideal as it disrupted other metrics


# In[275]:


import lightgbm as lgb


# In[276]:


model = ImbPipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('smote', SMOTE(random_state=42)),
    ('clf', lgb.LGBMClassifier(is_unbalance=True, random_state=42))
])


# In[277]:


get_ipython().run_cell_magic('time', '', '\nmodel.fit(X_train, y_train)')


# In[278]:


y_predict = model.predict(X_test)


# In[279]:


print(metrics.classification_report(y_test,y_predict))


# In[280]:


from catboost import CatBoostClassifier


# In[281]:


model = ImbPipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('smote', SMOTE(random_state=42)),
    ('clf', CatBoostClassifier(auto_class_weights='Balanced', random_state=42))
])


# In[282]:


get_ipython().run_cell_magic('time', '', '\nmodel.fit(X_train, y_train)')


# In[283]:


y_predict = model.predict(X_test)


# In[284]:


print(metrics.classification_report(y_test,y_predict))                              #### its not improving beyond this point


# In[285]:


cm = metrics.confusion_matrix(y_test,y_predict)

labels = ['Bad','OK','Good']

disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)

fig, ax = plt.subplots(figsize=(25,12))

disp.plot(ax=ax)


# In[ ]:


############################################################################################################################
#### We have concluded the modeling phase for the Netflix reviews analysis. Despite extensive efforts, the model's #########
#### performance has plateaued, primarily due to the imbalance in the target variable. To address this, we implemented #####
#### an imbalanced pipeline incorporating SMOTE, which helped mitigate the effects of the imbalance and improved the #######
#### model's performance. However, after thorough experimentation, further improvements have proven to be marginal #########
############################################################################################################################

