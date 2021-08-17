'''
Statistical method
Reduce number of parameter  base on variance
The main idea of this method is compute variance of all of numeric parameters and remove parameter if it's less than
a threshold
'''
'''
from sklearn.feature_selection import VarianceThreshold
from sklearn.datasets import make_classification

#Get example data from sklearn package

X,y = make_classification()

print('X: \n', X[:5, :5])
print('y: \n', y)
print('X shape: ', X.shape)
print('y shape: ', y.shape)

# theshold of variance = 0.5

VarianceThreshold(0.5).fit_transform(X).shape

#This dataset was not change since all of variance > 0.5
#If we increase value of threshold to 0.9

# VarianceThreshold(0.9).fit_transform(X).shape
'''
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import  cross_val_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.datasets import make_classification

#Get example data from sklearn package

X,y = make_classification()

X_kbest = SelectKBest(f_classif, k =5).fit_transform(X,y)
X_kvar = VarianceThreshold(0.9).fit_transform(X)

print('X shape after apllying statistical selection: ', X_kbest.shape)
print('X shape after aplly variance selection:', X_kvar.shape)


#Logistic Regression

logit = LogisticRegression(solver='lbfgs', random_state=1)

#Cross validation for:
#1. Original data
acc_org = cross_val_score(logit, X, y, scoring='accuracy', cv= 5).mean()
#2. Apply variance:
acc_var = cross_val_score(logit, X_kvar, y, scoring='accuracy', cv= 5).mean()
#3. Apply statistical method
acc_stat = cross_val_score(logit, X_kbest, y, scoring='accuracy', cv= 5).mean()

print('Accuracy on original data:', acc_org)
print('Accuracy on data applying variance:', acc_var)
print('Accuracy on data apllying statistical method:', acc_stat)