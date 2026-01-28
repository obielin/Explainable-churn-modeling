#!/usr/bin/env python
# coding: utf-8

# # Bank Customer Churn Analysis

# In[1]:


get_ipython().run_line_magic('pip', 'install kaggle')


# - Downloading and Extracting the Dataset
# 
# First, we download the dataset from Kaggle:

# In[2]:


get_ipython().system('kaggle datasets download -d "gauravtopre/bank-customer-churn-dataset"')


# This command downloads the dataset titled "Bank Customer Churn Dataset" from Kaggle.
# 
# Next, we extract the downloaded dataset:

# In[3]:


import zipfile
with zipfile.ZipFile("bank-customer-churn-dataset.zip", "r") as file:
    file.extractall("churn dataset")


# -  The dataset, which is in a ZIP file, is extracted into a folder named "churn dataset".

# In[4]:


import os
os.listdir("churn dataset")


# In[5]:


import pandas as pd
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# - To start the analysis, we load the data into a pandas DataFrame:
# This code reads the CSV file into a DataFrame df and displays the first five rows.

# In[6]:


df = pd.read_csv('churn dataset/Bank Customer Churn Prediction.csv')
df.head(5)


# ##  Initial Data Exploration
# General Information
# - We obtain general information about the dataset:

# In[7]:


#Get general info about customer dataset
df.shape
df.info()


# Analyzing the data types of each column:
# 
# - This code counts the occurrences of each data type in the DataFrame.

# In[8]:


#Get count analysis of datatypes in the customer dataset.
df.dtypes.value_counts()


# In[9]:


#To check for missing values
df.isna().sum()


# -   generate descriptive statistics that summarize the central tendency, dispersion, and shape of the dataset's distribution.

# In[10]:


#SUMMARY STATISTICS
df.describe()


# ###  EXPLORATORY DATA ANALYSIS

# - Count the number of instances in the dataset where customers have either exited or retained a service

# In[11]:


# Count the number of instances for 'exited' (1) and 'retained' (0)
num_retained = df['churn'].value_counts()[0]
num_exited = df['churn'].value_counts()[1]

print("Number of retained instances:", num_retained)
print("Number of exited instances:", num_exited)


# ####   Create a donut chart for visualizing the proportions of churned and retained customers in a dataset.

# In[12]:


# Define data
labels = ['Churned', 'Retained']
sizes = [df['churn'].sum(), df['churn'].count() - df['churn'].sum()]
total = sum(sizes)
percentages = [size/total * 100 for size in sizes]

# Create sunburst chart
plt.figure(figsize=(5, 4))

# Define the colors (using a more colorful palette)
colors = ['#ff9999','#66b3ff']  # Light Red and Light Blue

plt.pie(sizes, labels=labels, startangle=90, counterclock=False, autopct='%1.1f%%',
        colors=colors, wedgeprops={'edgecolor': 'white'})

# Add circle in the middle to create the donut chart
circle = plt.Circle((0, 0), 0.6, color='white')
plt.gca().add_artist(circle)

# Add a title
plt.title("Proportion of customers who churned and were retained", fontsize=16)

plt.axis('equal')

# Show plot
plt.show()


# - The doughnut chart visualizes the proportion of customers who churned versus those who were retained. The majority of the customer base, 79.6%, remained with the service or company, while 20.4% churned. 
# -   The high retention rate is positive, but the significant churn rate warrants investigation to minimize costs and implement targeted retention strategies.

# #### Generate  boxplots for numeric variables in the dataset, 
# - The boxplots provides  insights into the distribution of each variable, including the presence of outliers.
# 

# In[13]:


import matplotlib.pyplot as plt
import seaborn as sns

# List of numeric variables
numeric_vars = ['credit_score', 'age', 'tenure', 'balance', 'products_number', 'active_member', 'estimated_salary']

# List of colors for the boxplots
colors = ['skyblue', 'lightgreen', 'lightpink', 'lightcoral', 'lightskyblue']

for var, color in zip(numeric_vars, colors):
    plt.figure(figsize=(16, 6))
    sns.boxplot(x=var, data=df, color=color)  # Apply the color from the colors list
    plt.title(f"Boxplot of {var} showing outliers")
    plt.show()


# -   The boxplot for credit scores shows an even distribution across the middle 50% of data, with a median score around the 650 mark. Outliers indicate a small number of customers with significantly lower scores. The credit score outliers may represent a segment with potentially higher credit risks,
# -   The age distribution is narrow, with the median age around the late 30s to early 40s. The age outliers may indicate a broader age range of service appeal, and the product number distribution suggests most customers prefer fewer products.
# -   The product number distribution shows a concentration of data at the lower end, with the median at one product. Outliers at the higher end suggest very few customers use a large number of products. The lack of spread in the IQR and the position of the median suggest most customers are clustered around using a single product or service.  

# ####   Create a series of histograms for selected numeric variables, offering a graphical representation of their distribution and frequency.

# In[14]:


# Numeric variables to visualize outliers
numeric_vars = ['credit_score', 'age', 'balance', 'estimated_salary'] 		

# Colors for each histogram
colors = ['skyblue', 'salmon', 'gold', 'lightgreen', 'plum']

# Plot histograms for each numeric variable
for var, color in zip(numeric_vars, colors):
    plt.figure(figsize=(8, 6))
    plt.hist(df[var], bins=20, color=color, edgecolor='black', alpha=0.7)
    plt.title(f'Histogram of {var.capitalize()}', fontsize=16)  # Capitalize the variable name for the title
    plt.xlabel(var.capitalize(), fontsize=14)  # Capitalize the variable name for the x-axis label
    plt.ylabel('Frequency', fontsize=14)
    plt.tight_layout()
    plt.show()


# -   These histograms provide a visual summary of the key numerical data within the dataset. The credit score and age distributions suggest typical demographic patterns, while the balance histogram highlights a potential anomaly or unique feature in how customers are distributed by their account balance. The uniform distribution of estimated salaries may suggest that salary is not a variable with much variance in this particular dataset, or that the company serves a wide range of income levels equally.
# 
# - Credit Score:
#     -   The distribution is bell-shaped, which suggests a normal distribution of credit scores among the customers. Most scores center around the median, with fewer customers having very low or very high scores. The slight right skew indicates more customers have higher credit scores than lower.
# 
# - Age:
#     - The age distribution is also bell-shaped but with a slight right skew. This indicates that while most customers are middle-aged, there is a tail of older customers.
# Histogram of Balance:
#     -   This histogram shows a significant number of customers with a zero balance, which is distinctly separate from the rest of the distribution. The remaining balances appear to be roughly uniformly distributed, with fewer customers having very high balances.
# 
# -   Estimated Salary:
#     -   The salary distribution looks fairly uniform across the range, suggesting that salary may not be a distinguishing feature among the customers. There is no clear concentration of customers at any specific salary level, and it is interesting to note the consistency across all salary levels.

# - Generate dodge bar plots for different variables in a dataset, with a focus on comparing the counts of retained and churned customers across various categories

# In[15]:


# Define variables
vars = ['country', 'gender', 'credit_card', 'active_member', 'products_number'] 


# In[16]:


df['churn'] = df['churn'].astype(str)

# Plot using dodge bar plot
for var in vars:
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x=var, hue='churn', data=df, dodge=True, palette='viridis')
    ax.set_title(f'Number of Customers by {var}', fontsize=16)
    ax.set_xlabel(var.capitalize(), fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    plt.legend(title='churn', labels=['Churned', 'Retained'])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



# -   Number of Customers by Country:
#     -   The bar chart shows that the number of retained customers is higher than the number of churned customers in each country.
#     -   France has a noticeably higher number of retained customers compared to Germany and Spain, this might indicate regional differences in customer satisfaction.
#     -   The churn rate relative to the number of retained customers could also vary by country, indicating that some markets may be more competitive or have different customer expectations.
# 
# -  Number of Customers by Gender:
#     -   This chart shows a distribution of customer retention and churn by gender.
#     -   The significantly higher churn rate of the female class could suggest a need to tailor customer engagement strategies for that demographic.
#     -   Could also signify that gender is a significant factor in customer retention for the company's products or services. 
# 
# -   Number of Customers by Credit Card Ownership:
#     -   The visualization indicates that customers who own a credit card have a higher retention rate, which could imply that having a credit card is associated with a higher likelihood of staying with the company. However, we see a significant churn rate is seen for customers who own a credit card. 
#     -   Credit card ownership may reflect market saturation, making it easier to switch between service providers. Financial behaviour may also influence churn decisions. Competitive offers may also attract customers, particularly younger or higher-income customers. Service dissatisfaction could also contribute to higher churn rates.
# 
# -   Number of Customers by Active Membership:
#     -   The bar chart indicates that active members are less likely to churn compared to inactive ones. The Retained class is significant for active members, suggesting that active engagement with the company's services or products is associated with higher retention rates.  Conversely, the significant number of  churned inactive members highlights a potential area for improvement in engaging and retaining these customers.
# 
# -   Number of Customers by Products Number:
#     -   This bar chart shows the distribution of retained and churned customers based on the number of products they use. Most customers have two products and customers with one product are more likel to churn. It implies that customers engaged with more range of products tend to stay with the company. This can suggest customer loyalty increases with product variety or integration. 
#     -   On the other hand, a spike the Churned customers with three products could indicate points where customers feel overwhelmed or dissatisfied with the product offerings or where the bank might be losing customers to competitors.

# In[17]:


numeric_data = df.select_dtypes(include=[np.number])
categorical_data = df.select_dtypes(exclude=[np.number])


# In[18]:


numeric_data.head()


# In[19]:


categorical_data.head()


# In[20]:


churn_column = df['churn'].copy()


# -   The StandardScaler standardizes features by removing the mean and scaling to unit variance. This is particularly important when different features have different scales.
# - Many machine learning algorithms perform better or converge faster when features are on a relatively similar scale. Scaling can also help in reducing the impact of outliers.
# -   prepare the numeric data for machine learning models, ensuring that the features contribute equally to the model's performance and making algorithms less sensitive to the scale of features.

# In[21]:


from sklearn.preprocessing import StandardScaler

# Initialise the standard scalre and scale the data
scaler = StandardScaler()
scaled_numerical_data = scaler.fit_transform(numeric_data)

# convert the scaled data back to a DataFrame:
scaled_numerical_df = pd.DataFrame(scaled_numerical_data, columns=numeric_data.columns)


#  -  Perform one-hot encoding on categorical columns and then converts the resulting DataFrame to an integer type
#  -  Machine learning models generally require numerical input, so categorical data are often transformed into numbers. One-hot encoding is a standard method for this transformation

# In[22]:


# One-hot encode the categorical columns
encoded_categorical_df = pd.get_dummies(df[['country', 'gender']])

# Convert the DataFrame to integer type
encoded_categorical_df = encoded_categorical_df.astype(int)


# In[23]:


# Combine scaled numerical data and encoded categorical data
df_concat = pd.concat([scaled_numerical_df, encoded_categorical_df, churn_column], axis=1)


# In[24]:


df_concat.head()


#  -  Use of the Synthetic Minority Over-sampling Technique (SMOTE) to handle class imbalance in the dataset, followed by a train-test split. 

# In[25]:


from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from collections import Counter

# Assign the target variable
# Assign the target variable and convert to integer
y = df_concat["churn"].astype(int)
X = df_concat.drop('churn', axis=1)

# Instantiate the SMOTE oversampling technique
smote = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)

# Apply SMOTE to generate the oversampled dataset
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Check the new class distribution (optional)
print("Original class distribution:", Counter(y))
print("Resampled class distribution:", Counter(y_resampled))


# In[26]:


get_ipython().run_line_magic('pip', 'install lightgbm')
get_ipython().run_line_magic('pip', 'install xgboost')
get_ipython().run_line_magic('pip', 'install catboost')


# ## Model Selection (Champion Model)
# 
# The ensemble voting classifier is omitted due to estimator compatibility constraints across libraries.
# Instead, we select a single high-performing tree-based model ("champion model") and proceed with
# evaluation and SHAP-based explainability. This yields a more reproducible and interpretable workflow.

#  ####   Implemente a voting classifier using three popular gradient boosting frameworks: LightGBM, XGBoost, and CatBoost. 
#  -  The aim is to combine the predictive power of each individual model into a single, more robust classifier.

# In[28]:


from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Re-define the individual models (so the cell works even after a kernel restart)
lgbm = LGBMClassifier(
    objective='binary', boosting_type='gbdt', metric="auc",
    random_state=42, colsample_bytree=0.56, subsample=0.35,
    learning_rate=0.05, max_depth=8, n_estimators=500,
    num_leaves=140, reg_alpha=0.14, reg_lambda=0.85, verbosity=-1
)

xgb = XGBClassifier(
    objective='binary:logistic', eval_metric="auc", random_state=42,
    colsample_bytree=0.25, learning_rate=0.07, max_depth=8,
    n_estimators=800, reg_alpha=0.09, reg_lambda=0.70,
    min_child_weight=22, verbosity=0
)

cat = CatBoostClassifier(
    iterations=500, objective='Logloss', eval_metric="AUC",
    early_stopping_rounds=1000, bagging_temperature=0.1,
    colsample_bylevel=0.88, learning_rate=0.065, max_depth=7,
    l2_leaf_reg=1, min_data_in_leaf=25, random_strength=0.1,
    max_bin=100, verbose=0
)

models = {"LGBM": lgbm, "XGB": xgb, "CatBoost": cat}
scores = {}

for name, model in models.items():
    print(f"Fitting {name}...")
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    scores[name] = auc
    print(f"{name} ROC-AUC: {auc:.6f}\n")

best_name = max(scores, key=scores.get)
best_model = models[best_name]

print("Champion model:", best_name, "with ROC-AUC:", scores[best_name])


# ## Explainability (SHAP)
# 
# We use SHAP to quantify feature contributions:
# - Global: which features drive churn overall
# - Local: why a particular customer was predicted to churn
# 
# SHAP (SHapley Additive exPlanations) is used to:
# - identify global drivers of customer churn,
# - explain individual predictions,
# - support human-in-the-loop decision-making.

# In[29]:


import shap
import matplotlib.pyplot as plt

X_test_df = X_test.copy() 


# In[30]:


explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test_df)


# In[31]:


# For binary classification, SHAP may return a list [class0, class1]
if isinstance(shap_values, list):
    shap_values_pos = shap_values[1]  # class 1 = "churn"
else:
    shap_values_pos = shap_values


# In[32]:


shap.summary_plot(shap_values_pos, X_test_df, plot_type="bar")


# In[33]:


shap.summary_plot(shap_values_pos, X_test_df)


# In[34]:


i = 0
shap.force_plot(
    explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
    shap_values_pos[i],
    X_test_df.iloc[i],
    matplotlib=True
)


# In[36]:


get_ipython().system('jupyter nbconvert --to script "Customer churn prediction.ipynb"')


# In[ ]:




