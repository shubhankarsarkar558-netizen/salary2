import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load Dataset
df=pd.read_csv('salary_prediction.csv')
print(df)

# clean and explore
print(df.isnull().sum())
print(df.info())
print(df.describe())

#  LabelEncoder
lb=LabelEncoder()
df['Work_Mode']=lb.fit_transform(df['Work_Mode'])
df['Skills']=lb.fit_transform(df['Skills'])
print(df)

# Features Dataset
x=df.drop('Salary',axis=1)
y=df['Salary']
print(x)
print(y)

# train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)

# Prediction
y_pred=model.predict(x_test)

# Evaluation
rmse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("Rmse:",rmse)
print("R1 score:",r2)

# Save model
joblib.dump(model,"salary_prediction_model.pkl")
print("Model saved successfully")

# user input
Experience=float(input("Enter Experience:"))
Work_Mode=int(input("Work_Mode (office=0,work_from_home=1):"))
Working_Hours=int(input("Enter Working_Hours:"))
Projects=int(input("Enter number of Projects:"))
Skills=int(input("Skills (C++=0,Java=1,ML=2,Python=3):"))
Attendance=int(input("Enter Attendance:"))

#user data
user_data=[[Experience,Work_Mode,Working_Hours,Projects,Skills,Attendance]]

# predict salary
result=model.predict(user_data)
print("Predicted salary=",result[0])