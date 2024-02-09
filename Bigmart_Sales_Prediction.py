#!/usr/bin/env python
# coding: utf-8

# ## Big Mart Sales Prediction( Regression Model )
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


m = r'C:\Users\Pratik Sonawane\Downloads\Bigmart_sales.csv'
df = pd.read_csv(m)


# In[4]:


df.head()


# In[6]:


df.shape


# In[7]:


df.info()


# In[82]:


cat_col = df3.select_dtypes(include = ['object']).columns.tolist()
num_col = df3.select_dtypes(exclude = ['object']).columns.tolist()


# In[9]:


for col in cat_col:
    print(df[col].value_counts())
    print('---------------------------------------------')


# In[10]:


df.duplicated().sum()


# In[11]:


df.isnull().sum()


# In[12]:


df.groupby('Item_Identifier')['Item_Weight'].mean()


# In[13]:


df1 = df.copy()


# In[38]:


def fillIW(df2):
    item_avg_w = df2.groupby('Item_Identifier')['Item_Weight'].transform('mean')
    df2['Item_Weight'].fillna(item_avg_w,inplace=True)
    return df2


# In[40]:


df2 = fillIW(df2.copy())


# In[41]:


df2.isnull().sum()


# In[42]:


df2[df2['Item_Weight'].isnull()]


# the above 4 had unique item_identifier hence mean cannot be calculated so we are dropping it

# In[44]:


df3= df2.copy()


# In[50]:


def fillos(df3):
        mode_outlet_size = df3.groupby('Outlet_Type')['Outlet_Size'] \
                         .transform(lambda x: x.mode()[0] if x.mode().any() else pd.NA)
#If the group's Outlet_Size series has any mode (most frequent value)
#(x.mode().any()), it extracts the first mode using x.mode()[0]
        df3['Outlet_Size'] = df3['Outlet_Size'].fillna(mode_outlet_size)
        df3['Outlet_Size'] = df3['Outlet_Size'].fillna(df3['Outlet_Size'].mode()[0])

        return df3

df3 = fillos(df3.copy())  

print(df3['Outlet_Size'].isnull().sum())


# In[53]:


df3.groupby('Outlet_Type')['Outlet_Size'].value_counts()


# In[54]:


df3.isnull().sum()


# In[56]:


df3 = df3.dropna()


# In[57]:


df3.isnull().sum()


# In[58]:


df3.describe()


# min Item_Visibility is 0 we need to replace it with mean 

# In[61]:


# (.loc) to access the entire column (indicated by :), ensuring we only modify the intended column
df3.loc[:,'Item_Visibility'].replace([0],[df3['Item_Visibility'].mean()],inplace=True)


# In[62]:


df3.describe()


# In[63]:


df3['Item_Fat_Content'].value_counts()


# In[66]:


df3['Item_Fat_Content']=df3['Item_Fat_Content'].replace({'LF':'Low Fat','reg':'Regular','low fat':'Low Fat'})
df3['Item_Fat_Content'].value_counts()


# In[68]:


df3['Item_Identifier']


# In[74]:


df3['New_item_Types']= df3['Item_Identifier'].apply(lambda x : x[:2])
df3['New_item_Types']


# In[75]:


df3['New_item_Types'] = df3['New_item_Types'].map({'FD':'Food','NC':'Non Consumable','DR':'Drinks'})
df3['New_item_Types']


# In[76]:


df3['New_item_Types'].value_counts()


# In[81]:


df3.head()


# ## EDA

# In[83]:


for col in num_col:
    plt.hist(df3[col])
    plt.title(col)
    plt.show()


# num_col is not normally distributed

# In[85]:


df3.head()


# In[136]:


df3=df3.drop(columns = ['Item_Identifier'])


# In[93]:


plt.hist(df3['Item_Fat_Content'],bins = 3)
  


# In[120]:


count = df3['Item_Type'].value_counts()
plt.bar(count.index,count.values)
plt.xticks(rotation = 'vertical')


# In[123]:


count = df3['Outlet_Establishment_Year'].value_counts().sort_index()
plt.bar(count.index.astype(str),count.values)



# In[124]:


count = df3['Outlet_Size'].value_counts()
plt.bar(count.index,count.values)


# In[126]:


count = df3['Outlet_Location_Type'].value_counts()
plt.bar(count.index,count.values)


# In[128]:


count = df3['Outlet_Type'].value_counts()
plt.bar(count.index,count.values)
plt.xticks(rotation='vertical')


# In[130]:


# Check corr()

corr =df3.corr(numeric_only=True)


# In[133]:


sns.heatmap(corr,annot=True,cmap='Blues')


# highest corr indicated item_outlet_sales depends on item_MRP

# In[134]:


plt.scatter(df3['Item_MRP'],df3['Item_Outlet_Sales'])


# ### Label Encoding
# transfrom categorical columns in numerical code

# In[137]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[138]:


df3.head(1)


# In[139]:


cat_columns = ['Item_Fat_Content','Item_Type','Outlet_Identifier',
              'Outlet_Size','Outlet_Location_Type',
               'Outlet_Type','New_item_Types']


# In[140]:


for col in cat_columns:
    df3[col]= le.fit_transform(df3[col])


# In[141]:


df3.head()


# ### Feature Scaling

# In[143]:


X = df3.drop(columns = ['Item_Outlet_Sales'])
y = df3['Item_Outlet_Sales']


# In[146]:


X.head()


# In[148]:


fs= ['Item_Weight','Item_Visibility','Item_MRP','Outlet_Establishment_Year']


# In[165]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X[fs])
X[fs]= scaler.transform(X[fs])


# In[166]:


X.head()


# ### Split Data

# In[167]:


from sklearn.model_selection import train_test_split


# In[169]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state= 42)
X_train.shape,X_test.shape


# ### Model 

# In[170]:


from sklearn.linear_model import LinearRegression


# In[171]:


lr = LinearRegression()


# In[172]:


lr.fit(X_train,y_train)


# In[174]:


y_pred = lr.predict(X_test)
y_pred


# In[175]:


from sklearn.metrics import r2_score,mean_squared_error


# In[180]:


lrr2=r2_score(y_test,y_pred)
lrr2


# In[181]:


lrmse=mean_squared_error(y_test,y_pred)
lrmse


# In[194]:


from sklearn.model_selection import cross_val_score

lrcv_scores = cross_val_score(lr, X, y, cv=5, scoring="r2")

lrcv_mean=lrcv_scores.mean()


# In[195]:


print('***********Linear Regressor***********')
print(lrr2)
print(lrmse)
print(lrcv_mean)


# In[190]:


from sklearn.linear_model import Lasso

lasso = Lasso()
lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_test)

lasso_r2 = r2_score(y_test, y_pred)
lasso_mse = mean_squared_error(y_test, y_pred)

lasso_cv_scores = cross_val_score(lasso, X, y, cv=5, scoring="r2")
lasso_cv_mean = lasso_cv_scores.mean()
print('***********Lasso***********')
print(lasso_r2)
print(lasso_mse)
print(lasso_cv_mean)


# In[196]:


from sklearn.linear_model import Ridge

ridge = Ridge()
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)

ridge_r2 = r2_score(y_test, y_pred)
ridge_mse = mean_squared_error(y_test, y_pred)

ridge_cv_scores = cross_val_score(ridge, X, y, cv=5, scoring="r2")
ridge_cv_mean = ridge_cv_scores.mean()

print('***********Ridge Model***********')
print(ridge_r2)
print(ridge_mse)
print(ridge_cv_mean)


# In[198]:


from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

dt_r2 = r2_score(y_test, y_pred)
dt_mse = mean_squared_error(y_test, y_pred)

dt_cv_scores = cross_val_score(dt, X, y, cv=5, scoring="r2")
dt_cv_mean = dt_cv_scores.mean()

print('***********Decision Tree  Model***********')
print(dt_r2)
print(dt_mse)
print(dt_cv_mean)


# In[199]:


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

rf_r2 = r2_score(y_test, y_pred)
rf_mse = mean_squared_error(y_test, y_pred)

rf_cv_scores = cross_val_score(rf, X, y, cv=5, scoring="r2")
rf_cv_mean = rf_cv_scores.mean()

print('***********Random Forest  Model***********')
print(rf_r2)
print(rf_mse)
print(rf_cv_mean)


# In[200]:


from sklearn.ensemble import ExtraTreesRegressor

et = ExtraTreesRegressor()
et.fit(X_train, y_train)
y_pred = et.predict(X_test)

et_r2 = r2_score(y_test, y_pred)
et_mse = mean_squared_error(y_test, y_pred)

et_cv_scores = cross_val_score(et, X, y, cv=5, scoring="r2")
et_cv_mean = et_cv_scores.mean()

print('***********Extra Tree Model***********')
print(et_r2)
print(et_mse) 
print(et_cv_mean)


# In[201]:


from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor()
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

knn_r2 = r2_score(y_test, y_pred)
knn_mse = mean_squared_error(y_test, y_pred)

knn_cv_scores = cross_val_score(knn, X, y, cv=5, scoring="r2")
knn_cv_mean = knn_cv_scores.mean()

print('***********KNeighbours  Model***********')
print(knn_r2)
print(knn_mse) 
print(knn_cv_mean)


# In[205]:


from sklearn.svm import SVR

svr = SVR()
svr.fit(X_train, y_train)
y_pred = svr.predict(X_test)

svr_r2 = r2_score(y_test, y_pred)
svr_mse = mean_squared_error(y_test, y_pred)

svr_cv_scores = cross_val_score(svr, X, y, cv=5, scoring="r2")
svr_cv_mean = svr_cv_scores.mean()

print('***********SVR Model***********')
print(svr_r2)
print(svr_mse) 
print(svr_cv_mean) 


# **Random Forest Proved To Be Best Model**

# ### Feature Importance

# In[212]:


feature_importances = rf.feature_importances_
feature_names = X.columns
feature_importances, feature_names = zip(*sorted(zip(feature_importances, feature_names), reverse=False))

plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importances)
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")
plt.grid(axis="x", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()


# ### Model Predictive System

# In[213]:


X.columns


# In[215]:


X.head(1)


# In[223]:


new_item =np.array([
    float(input('Enter Item Weight =')),
    float(input("Enter Item Fat Content = ")),
    float(input("Enter Item Visibility = ")),
    int(input('Enter Item Type =')),
    float(input("Enter Item MRP = ")),  
    input("Enter Outlet Identifier = "),  
    int(input("Enter Outlet Establishment Year = ")),
    input("Enter Outlet Size = "),
    input("Enter Outlet Location Type = "),
    input("Enter Outlet Type = "),
    int(input("Enter New Item Types = "))])

new_item = new_item.reshape(1, -1)

predict = rf.predict(new_item)
print('Sales Prediction By Model =',predict[0])


# In[220]:


df3.describe()


# In[ ]:




