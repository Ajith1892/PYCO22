
import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st 

from PIL import Image
from datetime import datetime as dt





pickle_in1 = open("onehot.sav","rb")
onehot=pickle.load(pickle_in1)

pickle_in2 = open("minmax.sav","rb")
minmax=pickle.load(pickle_in2)

pickle_in3 = open("adaboost.sav","rb")
adaboost=pickle.load(pickle_in3)
#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def predict_note_authentication(df):
    
    
   
    prediction=adaboost.predict(df)
    print(prediction)
    return prediction



def main():
    #st.title("Fraud Detection")
    html_temp = """
    <div style="background-color:#4169E1;padding:10px">
    <h2 style="color:white;text-align:center;"> Fraud Detection System ML App </h2>
    </div>
    """
    
      
    st.markdown(html_temp,unsafe_allow_html=True)
    User_id = st.text_input('User_id',0)
    Signup_time = st.date_input(label='Enter the Signup Date:',value=(dt(year=2022,month=1,day=1,hour=0,minute=0)),
                                key='#date_range',
                                help="signup_time event")
    Purchase_time = st.date_input(label='Enter the purchase Date:',value=(dt(year=2022,month=1,day=5,hour=0,minute=0)),
                                key='#date_range1',
                                help="purchase_time event")
    
    Purchase_value = st.text_input('Purchase_value',56)
    Device_id = st.text_input('Device_id',0)
    Source = st.selectbox('Source',('SEO','Ads','Direct'))
    Browser = st.selectbox("Browser",('Chrome','FireFox','IE','Opera','Safari'))
    Sex = st.selectbox("sex",('M','F'))
    Age = st.text_input("Age",25)
    Ip_address = st.text_input("Ip_address",0)
    
    
    
    time_to_buy = (Purchase_time- Signup_time).days
    
    # prepare dataframe
    user_id1=pd.DataFrame({'user_id':User_id},index=[0])
    user_id1=pd.DataFrame({'user_id':User_id},index=[0])
    signup_time1=pd.DataFrame({'signup_time':Signup_time},index=[0])
    purchase_time1=pd.DataFrame({'purchase_time':Purchase_time},index=[0])
    purchase_value1=pd.DataFrame({'purchase_value':Purchase_value},index=[0])
    device_id1=pd.DataFrame({'device_id':Device_id},index=[0])
    source1=pd.DataFrame({'source':Source},index=[0])
    browser1=pd.DataFrame({'browser':Browser},index=[0])
    sex1=pd.DataFrame({'Sex':Sex},index=[0])
    age1=pd.DataFrame({'age':Age},index=[0])
    ip_address1=pd.DataFrame({'ip_address':Ip_address},index=[0])
    
    #conact dataframes
    df=pd.concat([user_id1,signup_time1,purchase_time1,purchase_value1,device_id1,source1,browser1,
                  sex1,age1,ip_address1],axis=1)
    
    #prepare the new dataframe time to buy and drop uneccesary columns
    
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    df['time_to_buy'] = (df['purchase_time'] - df['signup_time']).dt.days
    
    
    df.drop(labels=['user_id','signup_time','purchase_time','device_id','ip_address'],axis=1,inplace=True)
      
    
    #perform one hot encoding
    test_onehot= onehot.transform(df[['source','browser','sex']])
    test_onehot1=pd.DataFrame(test_onehot,columns=onehot.get_feature_names_out())
    #st.write(test_onehot1)
    df=df.reset_index(drop=True)
    df=pd.concat([df,test_onehot1],axis=1)
    df=df.drop(columns=['source','browser','sex'],axis=1)
    
    
    #normalization 
    
    df[['purchase_value','age','time_to_buy']]=minmax.transform(df[['purchase_value','age','time_to_buy']])

  
    
    
    result=""
    if st.button("Predict"):
        result=predict_note_authentication(df)
        
        if result==1:
            html_temp1 = """
    <div style="background-color:tomato;padding:2px">
    <h3 style="color:white;text-align:center;"> Its is a Fraud Transaction </h2>
    </div>
    """
            
            
    
      
            st.markdown(html_temp1,unsafe_allow_html=True)
        else:
            html_temp2 = """
    <div style="background-color:#008000;padding:2px">
    <h3 style="color:white;text-align:center;"> Its is a Genuine Transaction </h2>
    </div>
    """
            st.markdown(html_temp2,unsafe_allow_html=True)
    
      
            
            
    #st.success('The output is {}'.format(result))
   

if __name__=='__main__':
  
    main()
    
    
    