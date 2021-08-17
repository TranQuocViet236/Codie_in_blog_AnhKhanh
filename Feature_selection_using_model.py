'''
Usually, There are 2 model to  evaluate variable is Random Forset
or Linear Regression
'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

X, y = make_classification()
logit = LogisticRegression(solver='lbfgs', random_state=1)
 # Regression by RandomForest
rdFrt = RandomForestClassifier(n_estimators=10, random_state= 1)
# Regression by LinearSVC
lnSVC = LinearSVC(C= 0.01, penalty='l1', dual=False)
#Create a pipeline select variable from RandomForest model and Regression by Logit
pipe1 = make_pipeline(StandardScaler(), SelectFromModel(estimator=rdFrt), logit)
#Create a pipeline select variable from SVC model and Regression by Logit
pipe2 = make_pipeline(StandardScaler(), SelectFromModel(estimator=lnSVC), logit)

# Cross validate with :
#1, Logit model
acc_log = cross_val_score(logit, X, y, scoring = 'accuracy', cv=5 ).mean()
#2, RandomForest model
acc_rdf = cross_val_score(rdFrt, X, y, scoring = 'accuracy', cv=5 ).mean()
#3, Pipe1 model
acc_pip1 = cross_val_score(pipe1, X, y, scoring = 'accuracy', cv=5 ).mean()
#1, Pipe2 model
acc_pip2 = cross_val_score(pipe2, X, y, scoring = 'accuracy', cv=5 ).mean()

print('Accuracy theo logit:', acc_log)
print('Accuracy theo random forest:', acc_rdf)
print('Accuracy theo pipeline 1:', acc_pip1)
print('Accuracy theo pipeline 2:', acc_pip2)


'''
Sequential Feature Selection
'''

from mlxtend.feature_selection import SequentialFeatureSelector

selector  = SequentialFeatureSelector(logit,
                                      scoring='accuracy',
                                      verbose= 2,
                                      k_features=3,
                                      forward=False,
                                      n_jobs=-1)

selector.fit(X,y)