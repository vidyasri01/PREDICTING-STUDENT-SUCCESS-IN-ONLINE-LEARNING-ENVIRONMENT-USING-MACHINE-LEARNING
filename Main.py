from tkinter import messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import tkinter
import numpy as np
from tkinter import filedialog
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from math import sqrt


main = tkinter.Tk()
main.title("Predicting Student Success in Online Learning Environments using Machine Learning")
main.geometry("1300x1200")

global filename
global accuracy, precision, recall, fscore, rmse
global dataset
global X, Y, score
global score_X_train, score_X_test, score_y_train, score_y_test
global label_encoder, rf_rg, rf_cls, labels
global grade_X_train, grade_X_test, grade_y_train, grade_y_test

def uploadDataset():
    global filename, dataset
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,'Dataset loaded\n\n')
    dataset = pd.read_csv(filename,nrows=5000)
    dataset.drop(['code_module'], axis = 1,inplace=True)
    text.insert(END,str(dataset))
    label = dataset.groupby('final_result').size()
    label.plot(kind="bar")
    plt.xlabel('Grades')
    plt.ylabel('Number of Records')
    plt.xticks(rotation=90)
    plt.title("Grades Graph")
    plt.show()
            
def processDataset():
    global dataset, labels, X, Y
    global score_X_train, score_X_test, score_y_train, score_y_test
    global grade_X_train, grade_X_test, grade_y_train, grade_y_test
    global label_encoder, score
    text.delete('1.0', END)
    labels = np.unique(dataset['final_result'])
    label_encoder = []
    columns = dataset.columns
    types = dataset.dtypes.values
    for i in range(len(types)):
        name = types[i]
        if name == 'object': #finding column with object type
            le = LabelEncoder()
            dataset[columns[i]] = pd.Series(le.fit_transform(dataset[columns[i]].astype(str)))#encode all str columns to numeric
            label_encoder.append([columns[i], le])
    dataset.fillna(0, inplace = True)
    Y = dataset['final_result'].ravel()
    score = dataset['score'].ravel()
    dataset.drop(['final_result', 'score'], axis = 1,inplace=True)
    X = dataset.values
    text.insert(END,str(dataset)+"\n\n")
    grade_X_train, grade_X_test, grade_y_train, grade_y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
    score_X_train, score_X_test, score_y_train, score_y_test = train_test_split(X, score, test_size=0.2) #split dataset into train and test
    text.insert(END,"Dataset Train & Test Split\n\n")
    text.insert(END,"80% dataset size used for training : "+str(grade_X_train.shape)+"\n")
    text.insert(END,"20% dataset size used for testing  : "+str(grade_X_test.shape)+"\n")

def calculateMetrics(algorithm, predict, y_test, rmse_score):
    global accuracy, precision, recall, fscore, rmse
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    rmse.append(rmse_score)
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FSCORE    :  "+str(f)+"\n")
    text.insert(END,algorithm+" RMSE      :  "+str(rmse_score)+"\n\n")

def runRandomForest():
    text.delete('1.0', END)
    global accuracy, precision, recall, fscore, rmse
    global rf_rg, rf_cls
    global score_X_train, score_X_test, score_y_train, score_y_test
    global grade_X_train, grade_X_test, grade_y_train, grade_y_test
    accuracy = []
    precision = []
    recall = []
    fscore = []
    rmse = []

    rf_cls = RandomForestClassifier() #create Random Forest object
    rf_cls.fit(grade_X_train, grade_y_train)
    predict = rf_cls.predict(grade_X_test)
    
    rf_rg = RandomForestRegressor() #create Random Forest object
    rf_rg.fit(score_X_train, score_y_train)
    predicts = rf_rg.predict(score_X_test)
    mse_value = mean_squared_error(score_y_test, predicts)
    rmse_value = sqrt(mse_value)
    calculateMetrics("Random Forest", predict, grade_y_test, rmse_value)
    
def runGradient():
    global accuracy, precision, recall, fscore, rmse
    global score_X_train, score_X_test, score_y_train, score_y_test
    global grade_X_train, grade_X_test, grade_y_train, grade_y_test

    gb_cls = GradientBoostingClassifier() #create GradientBoostingClassifier object
    gb_cls.fit(grade_X_train, grade_y_train)
    predict = gb_cls.predict(grade_X_test)

    gb_rg = GradientBoostingRegressor() #create Random Forest object
    gb_rg.fit(score_X_train, score_y_train)
    predicts = gb_rg.predict(score_X_test)
    mse_value = mean_squared_error(score_y_test, predicts)
    rmse_value = sqrt(mse_value)
    calculateMetrics("Gradient Boosting", predict, grade_y_test, rmse_value)


def graph():
    global accuracy, precision, recall, fscore, rmse
    df = pd.DataFrame([['Random Forest','Precision',precision[0]],['Random Forest','Recall',recall[0]],['Random Forest','F1 Score',fscore[0]],['Random Forest','Accuracy',accuracy[0]],
                       ['Gradient Boosting','Precision',precision[1]],['Gradient Boosting','Recall',recall[1]],['Gradient Boosting','F1 Score',fscore[1]],['Gradient Boosting','Accuracy',accuracy[1]],
                      ],columns=['Algorithms','Performance Output','Value'])
    df.pivot("Algorithms", "Performance Output", "Value").plot(kind='bar')
    plt.show()


def predict():
    text.delete('1.0', END)
    global rf_rg, rf_cls, label_encoder, labels
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    test = pd.read_csv(filename)
    test.drop(['code_module'], axis = 1,inplace=True)
    testData = pd.read_csv(filename)
    testData.drop(['code_module','final_result', 'score'], axis = 1,inplace=True)
    testData = testData.values
    for i in range(len(label_encoder)):
        temp = label_encoder[i]
        name = temp[0]
        le = temp[1]
        test[name] = pd.Series(le.transform(test[name].astype(str)))#encode all str columns to numeric
    test.fillna(0, inplace = True)
    test.drop(['final_result', 'score'], axis = 1,inplace=True)
    test = test.values
    predict_score = rf_rg.predict(test)
    predict_grade = rf_cls.predict(test)
    for i in range(len(predict_score)):
        text.insert(END,"Test Data = "+str(testData[i])+"\n")
        if labels[predict_grade[i]] == 'Fail':
            predict_score[i] = predict_score[i] / 2
        text.insert(END,"Predicted Score = "+str(predict_score[i])+"\n")
        text.insert(END,"Predicted Grade = "+str(labels[predict_grade[i]])+"\n\n")
        

def close():
    main.destroy()

font = ('times', 16, 'bold')
title = Label(main, text='Predicting Student Success in Online Learning Environments using Machine Learning')
title.config(bg='chocolate', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
upload = Button(main, text="Upload OULAD Online Student Dataset", command=uploadDataset)
upload.place(x=700,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='lawn green', fg='dodger blue')  
pathlabel.config(font=font1)           
pathlabel.place(x=700,y=150)

processButton = Button(main, text="Preprocess Dataset", command=processDataset)
processButton.place(x=700,y=200)
processButton.config(font=font1)

rfButton = Button(main, text="Run Random Forest", command=runRandomForest)
rfButton.place(x=700,y=250)
rfButton.config(font=font1) 

gbButton = Button(main, text="Run Gradient Boosting", command=runGradient)
gbButton.place(x=700,y=300)
gbButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=700,y=350)
graphButton.config(font=font1)

predictButton = Button(main, text="Predict Grade & Score", command=predict)
predictButton.place(x=700,y=400)
predictButton.config(font=font1)

closeButton = Button(main, text="Exit", command=close)
closeButton.place(x=700,y=450)
closeButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)


main.config(bg='light salmon')
main.mainloop()
