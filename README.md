# Ex.No: 13 Machine Learning – Mini Project
### DATE: 22/4/2024
### REGISTER NUMBER : 212221060222
### AIM:
To write a program to train the classifier for Diabetes.
### Algorithm:
Step 1: Import packages. Step 2: Get the data.
Step 3: Split the data. 
Step 4: Scale the data. 
Step 5: Instantiate model. 
Step 6: Create Gradio Function. 
Step 7: Print Result.

### Program:
```
import numpy as np
import pandas as pd
pip install gradio
pip install typing-extensions --upgrade
pip install --upgrade typing
pip install typing-extensions --upgrade
import gradio as gr
data = pd.read_csv('/content/diabetes.csv')
data.head()
print(data.columns)
x = data.drop(['Outcome'], axis=1)
y = data['Outcome']
print(x[:5])
#split data
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test= train_test_split(x,y)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

from sklearn.neural_network import MLPClassifier
model = MLPClassifier(max_iter=1000, alpha=1)
model.fit(x_train, y_train)
print("Model Accuracy on training set:", model.score(x_train, y_train))
print("Model Accuracy on Test Set:", model.score(x_test, y_test))

def diabetes(Pregnancies, Glucose, Blood_Pressure, SkinThickness, Insulin, BMI,Diabetes_Pedigree, Age):
    x = np.array([Pregnancies,Glucose,Blood_Pressure,SkinThickness,Insulin,BMI,Diabetes_Pedigree,Age])
    prediction = model.predict(x.reshape(1, -1))
    if(prediction==0):
      return "NO"
    else:
      return "YES"

outputs = gr.Textbox()
app = gr.Interface(fn=diabetes, inputs=['number','number','number','number','number','number','number','number'], outputs=outputs,description="Detection of Diabeties")
app.launch(share=True)
```


### Output:
![WhatsApp Image 2024-05-10 at 11 49 37_c97a60be](https://github.com/Rajithxx/13th-exp-AI/assets/148357145/66355f4d-1a98-4322-822b-3393b9358577)
![WhatsApp Image 2024-05-10 at 11 49 40_57d12a12](https://github.com/Rajithxx/13th-exp-AI/assets/148357145/7068b824-3c8d-46ed-870f-b0d778459d29)
![WhatsApp Image 2024-05-10 at 11 51 15_1149aa2f](https://github.com/Rajithxx/13th-exp-AI/assets/148357145/99d367ee-ba49-4fd0-a005-8e4de9d24a51)

### Result:
Thus the system was trained successfully and the prediction was carried out.
