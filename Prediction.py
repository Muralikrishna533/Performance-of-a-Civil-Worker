import numpy as np
import pickle

# loading the saved model
loaded_model = pickle.load(open('model.pkl', 'rb'))


## Building a Predictive System 
input_data = (25,2,6,16,2520,1,187,98.1,10.49,18.0,69.40,6,2,7.2,1)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1) 

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not productive')
else:
  print('The person is productive')