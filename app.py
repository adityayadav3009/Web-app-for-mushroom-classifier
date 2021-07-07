from scipy.sparse.construct import random
from sklearn import metrics
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve ,precision_score, recall_score 

st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    st.title("MUSHROOM CLASSIFICATION")
    st.sidebar.title("Model & parameters")
    st.markdown("How would you like your MUSHROOM to be? Edible or Poisonous 🍄")
    st.sidebar.markdown("If you are a good mushroom ,you dont have to fear 👮‍♂️")

    


    # cache DATA
    @st.cache(persist= True)
    def load_data():
        df =pd.read_csv("mushrooms.csv")
        label =LabelEncoder()

        for col in df.columns :
            df[col] =label.fit_transform(df[col])
        
        return df 
    
    @st.cache(persist=True)
    def split(df) :
        y =df.type 
        x =df.drop(columns =["type"] )

        X_train , X_test , y_train ,y_test =train_test_split(x ,y , test_size =0.3 ,random_state =0)
        return X_train , X_test , y_train ,y_test 

    def plot_metrics(metrics_list) :
        if "Cofusion Matrix" in metrics_list :
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model ,X_test ,y_test ,display_labels=["edible" ,"poisonous"] ,cmap="Blues_r")
            st.pyplot()
        if "ROC Curve" in metrics_list :
            st.subheader("ROC curve")
            plot_roc_curve(model ,X_test ,y_test )
            st.pyplot()
        if "Precision Recall Curve" in metrics_list :
            st.subheader("Precision -Recall Curve")
            plot_precision_recall_curve(model ,X_test ,y_test )
            st.pyplot()
    
    
    
    
    df = load_data()
    X_train , X_test , y_train ,y_test = split(df)

    

    st.sidebar.subheader("What classifier do you want to choose today ?")
    clf =st.sidebar.selectbox("Classifier" ,("Support Vector Machine" ,"Logistic Regression" ,"Random Forest Classifier"))

    #SVM

    if clf == "Support Vector Machine" :
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter) " ,0.01,10.0 ,step=0.01 ,key="C")
        kernel =st.sidebar.radio("Kernel" ,("rbf","linear") ,key='kernel')
        gamma= st.sidebar.radio("Gamma (Kernel coefficient)",("scale","auto"),key="gamma")

        metrics =st.sidebar.multiselect("What metrics to plot ?" ,("Cofusion Matrix" ,"ROC Curve","Precision Recall Curve"))

        if st.sidebar.button("Classify" ,key="classify") :
            st.subheader("SUPPORT VECTOR MACHINE (SVM) RESULTS")
            model =SVC(C=C ,kernel=kernel ,gamma=gamma)
            model.fit(X_train ,y_train)
            accuracy = model.score(X_test ,y_test)
            y_pred = model.predict(X_test)
            st.write("Accuracy   : " ,(accuracy.round(4))*100 ," % ")
            st.write("Precision  : " ,precision_score(y_test ,y_pred ).round(4)*100 ," % ")
            st.write("Recall \t\t\t\t    : " ,recall_score(y_test ,y_pred ).round(4)*100 ," % ")
            plot_metrics(metrics)
    
    #Logistic Regression

    if clf == "Logistic Regression" :
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter) " ,0.01,10.0 ,step=0.01 ,key="C_LR")
        max_iter =st.sidebar.slider("Maximum iterations",100,1000,key ="max_LR")

        metrics =st.sidebar.multiselect("What metrics to plot ?" ,("Cofusion Matrix" ,"ROC Curve","Precision Recall Curve"))

        if st.sidebar.button("Classify" ,key="classify") :
            st.subheader("LOGISTIC REGRESSION RESULTS")
            model =LogisticRegression(C=C ,max_iter=max_iter)
            model.fit(X_train ,y_train)
            accuracy = model.score(X_test ,y_test)
            y_pred = model.predict(X_test)
            st.write("R_2 score  : " ,(accuracy.round(4))*100 ," % ")
            st.write("Precision  : " ,precision_score(y_test ,y_pred ).round(4)*100 ," % ")
            st.write("Recall    : " ,recall_score(y_test ,y_pred ).round(4)*100 ," % ")
            plot_metrics(metrics)

    if clf == "Random Forest Classifier" :
        st.sidebar.subheader("Model Hyperparameters")
        n_est =st.sidebar.slider("Trees in the forest" ,100 ,1000 ,key="n")
        max_depth =st.sidebar.number_input("Maximum depth of tree ",1,10 ,step=1,key="dep")
        boot = st.sidebar.radio("Bootstrap sample ?",("True","False") ,key='boot')

        metrics =st.sidebar.multiselect("What metrics to plot ?" ,("Cofusion Matrix" ,"ROC Curve","Precision Recall Curve"))

        if st.sidebar.button("Classify" ,key="classify") :
            st.subheader("RANDOM FOREST CLASSIFIER")
            model =RandomForestClassifier(n_estimators=n_est ,max_depth=max_depth ,bootstrap=boot ,n_jobs=-1)
            model.fit(X_train ,y_train)
            accuracy = model.score(X_test ,y_test)
            y_pred = model.predict(X_test)
            st.write("Accuracy   : " ,(accuracy.round(4))*100 ," % ")
            st.write("Precision  : " ,precision_score(y_test ,y_pred ).round(4)*100 ," % ")
            st.write("Recall    : " ,recall_score(y_test ,y_pred ).round(4)*100 ," % ")
            plot_metrics(metrics)



    if st.sidebar.checkbox("Show raw data" ,False) :                    #initialised with false if checked will return 1  
        st.subheader("Mushroom Data Set {Classification}")
        st.markdown("The data is taken from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/mushroom)" )
        st.write(df)

    

    

if __name__ == '__main__':
    main()


