from datetime import datetime, timedelta,date
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline 

import xgboost as xgb
from xgboost import plot_tree
from xgboost import plot_importance
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm
import statsmodels.formula.api as smf

import chart_studio.plotly as py
import plotly.offline as pyoff
import plotly.graph_objs as go


def main():

    # Import the dataset

    dataset = './data/TelcoChurn.csv'
    df_data = pd.read_csv(dataset)

    # Replacing class labels as numeric 0's and 1's

    df_data.loc[df_data.Churn=='No','Churn'] = 0 
    df_data.loc[df_data.Churn=='Yes','Churn'] = 1

    # Convert object to floating point

    df_data['TotalCharges'] = pd.to_numeric(df_data['TotalCharges'], errors='coerce')
    
    # Analyse data

    # print (df_data.groupby('gender').Churn.mean())
    # print (df_data.tenure.describe())
    
    # Preprocess NaN values in TotalCharges

    # print ("Number of NaN records in TotalCharges = " + str(len(df_data[pd.to_numeric(df_data['TotalCharges'], errors='coerce').isnull()])))
    df_data.loc[pd.to_numeric(df_data['TotalCharges'], errors='coerce').isnull(),'TotalCharges'] = np.nan
    df_data = df_data.dropna()
    df_data['TotalCharges'] = pd.to_numeric(df_data['TotalCharges'], errors='coerce')

    # Visualize the data 
    # Uncomment if visualization is needed
    
    # visualize_bar(df_data, 'gender')
    # visualize_scatter(df_data, 'tenure')

    # Get the number of clusters using the elbow method
    # Uncomment if visualization is needed

    # elbow_method(df_data, 'tenure')
    # elbow_method(df_data, 'MonthlyCharges')
    # elbow_method(df_data, 'TotalCharges')

    # # Observe the profile of clusters

    dataframe = create_clusters(df_data, 'tenure', 3)
    dataframe = create_clusters(df_data, 'MonthlyCharges', 3)
    dataframe = create_clusters(df_data, 'TotalCharges', 3)

    #create feature set and labels
    X = dataframe.drop(['Churn','customerID'],axis=1)
    y = dataframe.Churn

    #train and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=56)

    # Perform grid search to get the best possible algorithm with its hyperparameters.
    # Uncomment when needed

    # grid_search(X_train, y_train)

    # Prediction using XGBoost or Logistic Regression
    # Uncomment whichever model is needed based on the dataset.

    model_glm(dataframe)
    model_xgb(dataframe, X_train, X_test, y_train, y_test)
    # model_LR(dataframe,  X_train, X_test, y_train, y_test)

def elbow_method(dataframe, column_name):
    sse={}
    df_cluster = dataframe[[column_name]]
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df_cluster)
        df_cluster["clusters"] = kmeans.labels_
        sse[k] = kmeans.inertia_ 
    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of cluster")
    plt.title(column_name)
    plt.show()

def create_clusters(dataframe, column_name, k):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(dataframe[[column_name]])
    cluster_column_name = str(column_name).replace("'",'').title() + 'Cluster'
    dataframe[cluster_column_name] = kmeans.predict(dataframe[[column_name]])

    dataframe = order_cluster(cluster_column_name, column_name, dataframe,True)
    
    dataframe[cluster_column_name] = dataframe[cluster_column_name].replace({0:'Low',1:'Mid',2:'High'})

    # print (dataframe.groupby(cluster_column_name).tenure.describe())

    # Uncomment if visualization is needed
    # visualize_bar(dataframe, cluster_column_name)
    
    dataframe = label_encoding(dataframe, cluster_column_name)
    return dataframe

def order_cluster(cluster_field_name, target_field_name,df,ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    
    return (df_final)
   
def label_encoding(dataframe, cluster_column_name):
    le = LabelEncoder()
    dummy_columns = [] #array for multiple value columns

    for column in dataframe.columns:
        if dataframe[column].dtype == object and column != 'customerID':
            if dataframe[column].nunique() == 2:
                #apply Label Encoder for binary ones
                dataframe[column] = le.fit_transform(dataframe[column]) 
            else:
                dummy_columns.append(column)

    dataframe = pd.get_dummies(data = dataframe,columns = dummy_columns)
    return dataframe

def grid_search(features, target):
    pipe = Pipeline([("classifier", RandomForestClassifier())])

    # Create dictionary with candidate learning algorithms and their hyperparameters
    search_space = [
                    {"classifier": [LogisticRegression()],
                    "classifier__penalty": ['l2','l1'],
                    "classifier__C": np.logspace(0, 4, 10),
                    "classifier__fit_intercept":[True, False],
                    "classifier__solver":['saga','liblinear']
                    },
                    {"classifier": [LogisticRegression()],
                    "classifier__penalty": ['l2'],
                    "classifier__C": np.logspace(0, 4, 10),
                    "classifier__solver":['newton-cg','saga','sag','liblinear'], ##These solvers don't allow L1 penalty
                    "classifier__fit_intercept":[True, False]
                    },
                    {"classifier": [RandomForestClassifier()],
                    "classifier__n_estimators": [10, 100, 1000],
                    "classifier__max_depth":[5,8,15,25,30,None],
                    "classifier__min_samples_leaf":[1,2,5,10,15,100],
                    "classifier__max_leaf_nodes": [2, 5,10]
                    },
                    {"classifier": [xgb.XGBClassifier()],
                    "classifier__learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
                    "classifier__max_depth": [ 3, 4, 5, 6, 8, 10, 12, 15],
                    "classifier__min_child_weight": [ 1, 3, 5, 7 ],
                    "classifier__gamma": [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
                    "classifier__colsample_bytree": [ 0.3, 0.4, 0.5 , 0.7 ]}]

    # create a gridsearch of the pipeline, the fit the best model
    gridsearch = GridSearchCV(pipe, search_space, cv=5, verbose=0,n_jobs=-1) # Fit grid search
    best_model = gridsearch.fit(features, target)

    print(best_model.best_estimator_)
    print("The mean accuracy of the model is:",best_model.score(features, target))  

def model_glm(dataframe):

    all_columns = []
    for column in dataframe.columns:
        column = column.replace(" ", "_").replace("(", "_").replace(")", "_").replace("-", "_")
        all_columns.append(column)

    dataframe.columns = all_columns

    glm_columns = 'gender'

    for column in dataframe.columns:
        if column not in ['Churn','customerID','gender']:
            glm_columns = glm_columns + ' + ' + column

    glm_model = smf.glm(formula='Churn ~ {}'.format(glm_columns), data=dataframe, family=sm.families.Binomial())
    res = glm_model.fit()
    print(res.summary())

    print (np.exp(res.params))

def model_LR(dataframe, X_train, X_test, y_train, y_test):

    lr_model = LogisticRegression().fit(X_train, y_train)

    print('Accuracy of Logistic Regression classifier on training set: {:.2f}'
        .format(lr_model.score(X_train, y_train)))
    print('Accuracy of Logistic Regression classifier on test set: {:.2f}'
        .format(lr_model.score(X_test[X_train.columns], y_test)))

    y_pred = lr_model.predict(X_test)
    print(classification_report(y_test, y_pred))

    dataframe['proba'] = lr_model.predict_proba(dataframe[X_train.columns])[:,1]
    print (dataframe[['customerID', 'proba']].head())
    visualize_feature_importance(lr_model)

def model_xgb(dataframe, X_train, X_test, y_train, y_test):
    #create feature set and labels
    X = dataframe.drop(['Churn','customerID'],axis=1)
    y = dataframe.Churn

    #train and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=56)

    #building the model
    xgb_model = xgb.XGBClassifier(max_depth=5, learning_rate=0.08, objective= 'binary:logistic',n_jobs=-1).fit(X_train, y_train)
    xgb_model = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=0.5, gamma=0.2,
       learning_rate=0.05, max_delta_step=0, max_depth=4,
       min_child_weight=7, missing=None, n_estimators=100, n_jobs=1,
       nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=None, subsample=1, verbosity=1).fit(X_train, y_train)

    print('Accuracy of XGB classifier on training set: {:.2f}'
        .format(xgb_model.score(X_train, y_train)))
    print('Accuracy of XGB classifier on test set: {:.2f}'
        .format(xgb_model.score(X_test[X_train.columns], y_test)))

    y_pred = xgb_model.predict(X_test)
    print(classification_report(y_test, y_pred))

    dataframe['proba'] = xgb_model.predict_proba(dataframe[X_train.columns])[:,1]
    # print (dataframe[['customerID', 'proba']].head())
    dataframe[['customerID', 'proba']].sort_values(by=['proba'], ascending=False).to_csv('churn_probability_1.csv')

    # Visualize the important features. Uncomment when needed
    # visualize_feature_importance(xgb_model)

def visualize_bar(dataframe, column_name):
    df_plot = dataframe.groupby(column_name).Churn.mean().reset_index() 
    plot_data = [
        go.Bar(
            x=df_plot[column_name],
            y=df_plot['Churn'],
            width = [0.5, 0.5],
            marker=dict(
            color=['green', 'blue', 'red', 'yellow'])
        )
    ]

    plot_layout = go.Layout(
            xaxis={"type": "category"},
            yaxis={"title": "Churn Rate"},
            title=column_name,
            plot_bgcolor  = 'rgb(243,243,243)',
            paper_bgcolor  = 'rgb(243,243,243)',
        )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    pyoff.iplot(fig)

def visualize_feature_importance(model):
    ##set up the parameters
    # fig, ax = plt.subplots(figsize=(100,100))
    # plot_tree(model)

    fig, ax = plt.subplots(figsize=(10,8))
    plot_importance(model, ax=ax)
    plt.show()

def visualize_scatter(dataframe, column):
    df_plot = dataframe.groupby('tenure').Churn.mean().reset_index()
    plot_data = [
    go.Scatter(
        x=df_plot['tenure'],
        y=df_plot['Churn'],
        mode='markers',
        name='Low',
        marker= dict(size= 7,
            line= dict(width=1),
            color= 'blue',
            opacity= 0.8
           ),
    )
    ]
    plot_layout = go.Layout(
            yaxis= {'title': "Churn Rate"},
            xaxis= {'title': "Tenure"},
            title='Tenure based Churn rate',
            plot_bgcolor  = "rgb(243,243,243)",
            paper_bgcolor  = "rgb(243,243,243)",
        )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    pyoff.iplot(fig)


if __name__ == "__main__":
    main()
