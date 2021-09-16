import pandas as pd
from sklearn import linear_model
import tkinter as tk 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.model_selection import cross_val_score
import numpy as np

import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('water_data.csv')
data['targetF'] = data.apply(lambda row: row['ExchemGF'] - row['experimental_HFE'], axis=1)
data.head()
data = data.values
X = data[:,1:2]
y = data[:,3]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.80, random_state = 1)

# with sklearn
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

print(regr)
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

predictions = regr.predict(X_test)
print(predictions)
r = np.corrcoef(predictions, y_test)
print(r)

kfold = KFold(n_splits = 50, random_state = 7)
results = cross_val_score(regr,X,y,cv = kfold)
print(results)
#print('Accuracy:' % (results.mean()*100.0, results.std()*100.0))

loocv = LeaveOneOut()
# enumerate splits
for train, test in loocv.split(data):
    print('train: %s, test: %s' % (data[train], data[test]))

# tkinter GUI
root= tk.Tk()

canvas1 = tk.Canvas(root, width = 500, height = 300)
canvas1.pack()

# with sklearn
Intercept_result = ('Intercept: ', regr.intercept_)
label_Intercept = tk.Label(root, text=Intercept_result, justify = 'center')
canvas1.create_window(260, 220, window=label_Intercept)

# with sklearn
Coefficients_result  = ('Coefficients: ', regr.coef_)
label_Coefficients = tk.Label(root, text=Coefficients_result, justify = 'center')
canvas1.create_window(260, 240, window=label_Coefficients)


# New GF-Excess Chemical Potential
label1 = tk.Label(root, text='Insert GF-Excess Chemical Potential: ')
canvas1.create_window(100, 100, window=label1)
entry1 = tk.Entry (root) # create 1st entry box
canvas1.create_window(270, 100, window=entry1)

# New PMV
label2 = tk.Label(root, text='Insert Partial Molar Volume: ')
canvas1.create_window(120, 120, window=label2)
entry2 = tk.Entry (root) # create 2nd entry box
canvas1.create_window(270, 120, window=entry2)

def values():
    global New_GF_Exchem
    New_GF_Exchem = float(entry1.get())

    global New_PMV
    New_PMV = float(entry2.get())
    
    New_prediction = regr.predict([[ New_PMV]])
    Predicted_HFE = New_GF_Exchem + New_prediction 

    Prediction_result  = ('Predicted Hydration Energy: ', Predicted_HFE)
    label_Prediction = tk.Label(root, text= Prediction_result, bg='orange')
    canvas1.create_window(260, 280, window=label_Prediction)
 
 
button1 = tk.Button (root, text='Predict Solvation Free Energy',command=values, bg='orange')  
canvas1.create_window(270, 150, window=button1)
 
#plot 1st scatter 
figure3 = plt.Figure(figsize=(5,4), dpi=100)
ax3 = figure3.add_subplot(111)
ax3.scatter(data[:,0].astype(float),data[:,2].astype(float), color = 'r')
scatter3 = FigureCanvasTkAgg(figure3, root) 
scatter3.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
ax3.legend(['experimental_HFE']) 
ax3.set_xlabel('ExchemGF')
ax3.set_title('GF-Excess Chemical Potential Vs. experimental_HFE')

#plot 2nd scatter 
figure4 = plt.Figure(figsize=(5,4), dpi=100)
ax4 = figure4.add_subplot(111)
ax4.scatter(data[:,1].astype(float),data[:,2].astype(float), color = 'g')
scatter4 = FigureCanvasTkAgg(figure4, root) 
scatter4.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
ax4.legend(['experimental_HFE']) 
ax4.set_xlabel('PMV')
ax4.set_title('Partial Molar Volume vs. experimental_HFE')

root.mainloop()
