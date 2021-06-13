import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import tree
import pickle

df = pd.read_csv("trainingdata.csv")


inputs = df.drop('Output',axis='columns')
target = df['Output']
le_DegreeofConcentration = LabelEncoder()
le_Academicperformance = LabelEncoder()
le_Flightfightresponse = LabelEncoder()
le_Sleepwakedisorder = LabelEncoder()
le_coronaphobia = LabelEncoder()

inputs['DegreeofConcentration_n'] = le_DegreeofConcentration.fit_transform(inputs['DegreeofConcentration'])
inputs['Academicperformance_n'] = le_Academicperformance.fit_transform(inputs['Academicperformance'])
inputs['Flightfightresponse_n'] = le_Flightfightresponse.fit_transform(inputs['Flightfightresponse'])
inputs['Sleepwakedisorder_n'] = le_Sleepwakedisorder.fit_transform(inputs['Sleepwakedisorder'])
inputs['coronaphobia_n'] = le_coronaphobia.fit_transform(inputs['coronaphobia'])

inputs_n = inputs.drop(["DegreeofConcentration","Academicperformance","Flightfightresponse","Sleepwakedisorder","coronaphobia"],axis='columns')

inputs_n_train, inputs_n_test, target_train, target_test = train_test_split(inputs_n, target, test_size=0.3, random_state=1) # 70% training and 30% test
#print(inputs_n)
x=pd.set_option("display.max_rows", None, "display.max_columns", None)
#print(x)

regressor = tree.DecisionTreeClassifier()

regressor.fit(inputs_n, target)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

print(model.predict([[1,0,0,1,1]]))



