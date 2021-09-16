import pandas as pd
from sklearn import linear_model
import tkinter as tk
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.utils import resample # for Bootstrap sampling
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import linear_model
#K-Fold CV
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.model_selection import cross_val_score
import sklearn.linear_model as skl_lm
import numpy as np
import warnings
from sklearn.utils import resample
import statsmodels.formula.api as sm

#%matplotlib inline

warnings.filterwarnings('ignore')

data = pd.read_csv('water_data.csv')

data['targetF'] = data.apply(lambda row: row['ExchemGF'] - row['experimental_HFE'], axis=1)
print(data.head())
data = data.values
X = data[:,1:2]
y = data[:,3]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.80, random_state = 1)

lm = skl_lm.LinearRegression()
model = lm.fit(X_train, y_train)
pred = model.predict(X_test)
pred2 = model.predict(X)

from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(y_test, pred)
print(MSE)
print("Intercept: ", model.intercept_)
print("slope: ", model.coef_)

# plot results
plt.scatter(X, y)
plt.plot(X, pred2, linewidth=2)
plt.grid(True)
plt.xlabel('Target Fn')
plt.ylabel('Predicted')
plt.title('Target vs Predicted')
plt.show()




from sklearn.model_selection import cross_val_score
loo = LeaveOneOut()
loo.get_n_splits(X)

from sklearn.model_selection import KFold

crossvalidation = KFold(n_splits=392, random_state=None, shuffle=False)

scores = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=crossvalidation, n_jobs=1)

print("Folds: " + str(len(scores)) + ", MSE: " + str(np.mean(np.abs(scores))) + ", STD: " + str(np.std(scores)))

#Resample
Xsamp, ysamp = resample(X, y, n_samples=500)
clf = model.fit(Xsamp,ysamp)
print('Intercept: ' + str(clf.intercept_) + " Coef: " + str(clf.coef_))



