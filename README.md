# UCEC_Recurrance
  this got overfitting but not 100% negative prediction (used k = 3)
Overall best params: {'clf__max_depth': 5, 'clf__min_samples_split': 4, 'clf__n_estimators': 50, 'select__k': 100} 

  this got 100% Negative prediction (used k = 7)
Overall best params: {'clf__max_depth': 10, 'clf__min_samples_split': 2, 'clf__n_estimators': 50, 'select__k': 1000} 

  this got AUC-ROC of 0.575 (using k = 3)
Overall best params: {'clf__max_depth': 5, 'clf__max_features': 'log2', 'clf__min_samples_split': 2, 'clf__n_estimators': 100, 'select__k': 100} 


Overall best params: {'clf__max_depth': 5, 'clf__max_features': 'log2', 'clf__min_samples_leaf': 2, 'clf__min_samples_split': 2, 'clf__n_estimators': 100, 'select__k': 100} 

  this what I got when I split 40% of data into the test, this one did really poorly (AUC-ROC = 0.49)
Overall best params: {'clf__max_depth': 10, 'clf__max_features': 0.1, 'clf__min_samples_leaf': 4, 'clf__min_samples_split': 2, 'clf__n_estimators': 50, 'select__k': 500} 

FOR LOGISTIC REGRESSION:
  AUC-ROC of 0.69
Overall best params: {'clf__C': 0.01, 'select__k': 50} 

for SVC (no LASSO):
Overall best params: {'classifier__C': 0.1, 'classifier__gamma': 'scale', 'classifier__kernel': 'rbf', 'select__k': 500} 
Mean AUC-ROC: 0.6541
Standard Deviation: 0.0140
BUT, no testing data got: 

True Positives (TP): 14
False Positives (FP): 35
True Negatives (TN): 47
False Negatives (FN): 3
AUC-ROC Score: 0.6984

