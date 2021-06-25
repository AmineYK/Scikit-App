import streamlit as st
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
from PIL import Image
import altair as alt
import pickle 

first,last = st.beta_columns([1,3])

img=Image.open("data/Scikit-app.png")
img1=Image.open("data/logo_white_large1.png")
st.sidebar.image(img1)
first.image(img)
last.title("""
         Scikit-App : Application for Stroke Predicition
         Cette Application vous donne accés à differents modéles de **Machine Learning** entrainés dans le but d'éffectuer des prédictions sur des patients par rapport au risque d'avoir un **Accident Vasculaire Celebrale AVC** ou pas . 
         """)
video_file = open('data/AVC.mp4', 'rb')
video_bytes = video_file.read()
st.video(video_bytes)
st.sidebar.title("Vos Informations")
model = st.sidebar.selectbox("Selectionner le modele de votre choix",("Logicstic Regression","Random Forest Classifier","Decision Tree Classifier","KNeighbors Classifier","SVC"))     
gender = st.sidebar.selectbox("Selectionner votre sexe",("Male","Female")) 
age = st.sidebar.slider("Selectionner votre age :",1,130)
hypertension = st.sidebar.radio("Avez-vous de l'hypertension ?",("Yes","No"))
heart_dis = st.sidebar.radio("Avez-vous de(s) maladie(s) cardiaque(s) ?",("Yes","No"))
#ever_married = st.sidebar.radio("Have you have been ever married ?",("Yes","No"))
work_type = st.sidebar.selectbox("Selectionner votre type de travail",("Private","Self-employed","Govt_job","children","Never_worked"))  
residence_type = st.sidebar.selectbox("Selectionner votre type de residence",("Rural","Urban")) 
glucose = st.sidebar.slider("Selectionner votre taux moyen de glucose dans le sang en (mg/dl) :",55,272)
bmi = st.sidebar.slider("Selectionner votre Indice de Masse Corporelle IMC:",10,50)
smoking_status = st.sidebar.selectbox("Selectionner votre status de fumeur",("never smoked","formerly smoked","smokes"))
confirm = st.sidebar.button("Voyons le resultat")
primaryColor = st.get_option("theme.primaryColor")
s = f"""
<style>
div.stButton > button:first-child {{ border: 3px outset {primaryColor}; border-radius:20px 20px 20px 20px;background-color:#C5C4C4;margin-left: 58px;margin-top:30px;}}
<style>
"""
st.markdown(s, unsafe_allow_html=True) 
avc_data = pd.read_csv("data/healthcare-dataset-stroke-data.csv",index_col='id')

st.subheader('Informations sur le dataset')

def encoder():
    
    if hypertension=="Yes":
       hypertension_num=1
    elif hypertension=="No":
       hypertension_num=0    
    
    if heart_dis=="Yes":
        heart_dis_num=1
    elif heart_dis=="No":
        heart_dis_num=0     
   
    if gender=="Male":
        gender_num=0
    elif gender=="Female":
        gender_num=1
     
    if work_type=="Private":
        work_type_num=0
    elif work_type=="Self-employed":
        work_type_num=1
    elif work_type=="Govt_job":
        work_type_num=2
    elif work_type=="children":
        work_type_num=3
    elif work_type=="Never_worked":
        work_type_num=4
    
    if residence_type=="Urban":
        residence_type_num=1
    elif residence_type=="Rural":
        residence_type_num=0    
        
    if smoking_status=="never smoked":
        smoking_status_num=0
    elif smoking_status=="smokes":
        smoking_status_num=1
    elif smoking_status=="formerly smoked":
        smoking_status_num=2
    elif smoking_status=="Unknown":
        smoking_status_num=3
    donnees_dict = {'gender':gender_num,
           'age':age,
           'hypertension':hypertension_num
           ,'heart_dis':heart_dis_num,
           'work_type':work_type_num,
           'residence_type':residence_type_num,
           'glucose':glucose,
           'bmi':bmi,
           'smoking_status':smoking_status_num
           }
    donnees = pd.DataFrame(donnees_dict,index=[0]) 
    return donnees
 
donnees = encoder()         
    
   
view = st.checkbox("Visualiser le dataset",True)

@st.cache(suppress_st_warning=True)
def voir_dataset():  
    if(view):
        st.write("Taille du Dataset : ",avc_data.shape)
        nom = "Stroke Prediction Dataset"
        st.write("Nom du Dataset : " ,nom)
        st.write(avc_data)
        st.table(avc_data.describe())
        st.write("Informations de l'utilisateur :")
        st.write(donnees)
    graphs = st.checkbox("Visualiser les graphes",True)
    if(graphs):
        col = st.selectbox("Choisissez une variable",avc_data.columns)
        chart = alt.Chart(avc_data).mark_bar().encode(
            x=col,
            y="count()",
            ).interactive()
        st.altair_chart(chart,use_container_width=True)
        st.line_chart(avc_data)
        


voir_dataset() 

def correlation(data,seuil):
    col_corr = set()
    corr_matrix = data.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j])>seuil : 
                colname=corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr


def nettoyer(avc_data):
    avc_data.fillna(value={'bmi':avc_data['bmi'].mean()},inplace=True)
    
    avc_data.replace('Other','Male',inplace=True)
   
    df1 = avc_data[avc_data['smoking_status'] == 'Unknown'].head(340)
    df1["smoking_status"].replace({"Unknown":"smokes"},inplace=True)
    avc_data.drop(avc_data.loc[avc_data['smoking_status']=='Unknown'].head(340).index, inplace=True)
    
    df2 = avc_data[avc_data['smoking_status'] == 'Unknown'].head(386)
    df2["smoking_status"].replace({"Unknown":"formerly smoked"},inplace=True)
    avc_data.drop(avc_data.loc[avc_data['smoking_status']=='Unknown'].head(386).index, inplace=True)
    
    df3 = avc_data[avc_data['smoking_status'] == 'Unknown'].head(818)
    df3["smoking_status"].replace({"Unknown":"never smoked"},inplace=True)
    avc_data.drop(avc_data.loc[avc_data['smoking_status']=='Unknown'].head(818).index, inplace=True)
 
    df4 = pd.concat([df2,df1])
    
    df4 = pd.concat([df4,df3])
    
    avc_data = pd.concat([avc_data,df4])
    
    corr_features = correlation(avc_data,0.6)
    avc_data = avc_data.drop("ever_married" ,axis=1)
    st.write(avc_data.shape)
    nom = "Stroke Prediction Dataset"
    st.write("Nom du Dataset : " ,nom)
    st.write(avc_data)
    
    
    
    
st.markdown("---")   
view2 = st.checkbox("Visualiser le dataset apres nettoyage",True)
     
if(view2):
    nettoyer(avc_data)
        
   
   
  

     
    
    
    
    
    
    
    
    
    
avc_data["gender"].replace({"Female":"0","Male":"1","Other":"2"},inplace=True)
avc_data["ever_married"].replace({"No":"0","Yes":"1"},inplace=True)
avc_data["Residence_type"].replace({"Rural":"0","Urban":"1"},inplace=True)
avc_data["smoking_status"].replace({"never smoked":"0","Unknown":"1","formerly smoked":"2","smokes":"3"},inplace=True)
avc_data["work_type"].replace({"Private":"0","Self-employed":"1","Govt_job":"2","children":"3","Never_worked":"4"},inplace=True)
avc_data = avc_data.astype({"gender":"int64","ever_married":"int64","work_type":"int64","Residence_type":"int64","smoking_status":"int64"})





seed = 96 
X = avc_data.drop(['stroke'],axis=1)
Y = avc_data['stroke']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=seed)




def LR_algo():
    lr = pickle.load(open("data/Regression logistique Finale.pk1","rb"))
    if lr.predict(donnees) == 0:
        st.sidebar.success("Bonne nouvelle : vous n'etes pas succeptible d'avoir un AVC :smile:")
    elif lr.predict(donnees) == 1:
        st.sidebar.warning("Mauvaise nouvelle : vous etes succeptible d'avoir un AVC 😬")
        
        
       
def RF_algo():
    rf = pickle.load(open("data/Forêt aléatoire Finale.pk1","rb"))
    if rf.predict(donnees) == 0:
        st.sidebar.success("Bonne nouvelle : vous n'etes pas succeptible d'avoir un AVC :smile:")
    elif rf.predict(donnees) == 1:
        st.sidebar.warning("Mauvaise nouvelle : vous etes succeptible d'avoir un AVC 😬")
        
        
        

       
def DT_algo():
    dt = pickle.load(open("data/Arbre décisionnel Finale.pk1","rb"))
    if dt.predict(donnees) == 0:
        st.sidebar.success("Bonne nouvelle : vous n'etes pas succeptible d'avoir un AVC :smile:")
    elif dt.predict(donnees) == 1:
        st.sidebar.warning("Mauvaise nouvelle : vous etes succeptible d'avoir un AVC 😬")
        
        
        
def KN_algo():
    kn = pickle.load(open("data/K-Plus proche voisin Finale.pk1","rb"))
    if kn.predict(donnees) == 0:
        st.sidebar.success("Bonne nouvelle : vous n'etes pas succeptible d'avoir un AVC :smile:")
    elif kn.predict(donnees) == 1:
        st.sidebar.warning("Mauvaise nouvelle : vous etes succeptible d'avoir un AVC 😬")
        
        

def SV_algo():
    sv = pickle.load(open("data/Machine à vecteur support Finale.pk1","rb"))
    if sv.predict(donnees) == 0:
        st.sidebar.success("Bonne nouvelle : vous n'etes pas succeptible d'avoir un AVC :smile:")
    elif sv.predict(donnees) == 1:
        st.sidebar.warning("Mauvaise nouvelle : vous etes succeptible d'avoir un AVC 😬")
        
        
        
if confirm:
   if model=="Logicstic Regression":
        LR_algo()
   elif model=="Random Forest Classifier":
        RF_algo()
   elif model=="Decision Tree Classifier":
        DT_algo()
   elif model=="KNeighbors Classifier":
        KN_algo()
   elif model=="SVC":
        SV_algo()