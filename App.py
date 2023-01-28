import numpy as np
import pickle
import streamlit as st


#loading the saved model
loaded_model = pickle.load(open('model.pkl', 'rb'))


#Creating a function for Prediction

def Labour_Prediction(Age, Designation, Experience, WorkingHours, RemainingWorkingHours, OTHours, HeartBeat, BodyTemperature, SkinSensor, RR, BloodVolume, MotionProductivity, MotionIndicater, ProductivityTime, NoiceDetection): 
    input_data = [[Age, Designation, Experience, WorkingHours, RemainingWorkingHours, OTHours, HeartBeat, BodyTemperature, SkinSensor, RR, BloodVolume, MotionProductivity, MotionIndicater, ProductivityTime, NoiceDetection]]

    

    # changing the input_data to numpy array
    #input_data_as_numpy_array = np.asarray(input_data)
  
    # reshape the array as we are predicting for one instance
    #input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)()

    prediction = loaded_model.predict(input_data)
    print(prediction)

    if (prediction[0] == 0):
     return 'The person is not productive'
    else:
     return 'The person is productive'
 
    
 
def main():
    
    #Giving a title
    st.title('Labour Performance Web App')
    
    
    #getting the input data from user
 
    Age = st.text_input('Various Age Group')
    Designation = st.text_input('Designation')
    Experience = st.text_input('Work_Experience')
    WorkingHours = st.text_input('Worked_Time')
    RemainingWorkingHours = st.text_input('Remaining Time')
    OTHours = st.text_input('Extra Time Worked')    
    HeartBeat = st.text_input('Heart Beat')
    BodyTemperature = st.text_input('Temperature')
    SkinSensor = st.text_input('Skin')
    RR = st.text_input('Respiration Rate')
    BloodVolume = st.text_input('Blood Volume')
    MotionProductivity = st.text_input('Motion Productivity')
    MotionIndicater = st.text_input('Motion Indicator')
    ProductivityTime =  st.text_input('Production Time')
    NoiceDetection = st.text_input('Noise Alert')
    
    
    
    #Codes for Prediction
    Productivity = ''
    
    #Creating a button for prediction
    if st.button('Productivity Test result'):
        Productivity = Labour_Prediction(Age, Designation, Experience, WorkingHours, RemainingWorkingHours, OTHours, HeartBeat, BodyTemperature, SkinSensor, RR, BloodVolume, MotionProductivity, MotionIndicater, ProductivityTime, NoiceDetection )
    
    st.success(Productivity)
    
    
    
if __name__ == "__main__":
    main()

