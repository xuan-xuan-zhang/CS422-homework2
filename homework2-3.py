import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import numpy as np
#Data loading and preprocessing
url="https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
col_names=['ID number', 'diagnosis']+[f'feature_{i}' for i in range(1,31)]
df = pd.read_csv(url,names=col_names)
y = df['diagnosis'].map({'M': 1, 'B': 0})
X = df.drop(['ID number', 'diagnosis'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#Create a decision tree classifier
dtc = DecisionTreeClassifier(min_samples_leaf=2, min_samples_split=5, max_depth=2, random_state=42)
#Perform dimensionality reduction using PCA and train the model.
pca_1 = PCA(n_components=1)
X_train_pca_1 = pca_1.fit_transform(X_train)
X_test_pca_1 = pca_1.transform(X_test)
dtc.fit(X_train_pca_1, y_train)
y_pred_pca_1 = dtc.predict(X_test_pca_1)
f1_pca_1 = f1_score(y_test, y_pred_pca_1)
precision_pca_1 = precision_score(y_test, y_pred_pca_1)
recall_pca_1 = recall_score(y_test, y_pred_pca_1)
#Perform dimensionality reduction using PCA and train the model.
pca_2 = PCA(n_components=2)
X_train_pca_2 = pca_2.fit_transform(X_train)
X_test_pca_2 = pca_2.transform(X_test)
dtc.fit(X_train_pca_2, y_train)
y_pred_pca_2 = dtc.predict(X_test_pca_2)
f1_pca_2 = f1_score(y_test, y_pred_pca_2)
precision_pca_2 = precision_score(y_test, y_pred_pca_2)
recall_pca_2 = recall_score(y_test, y_pred_pca_2)
#Train the model using the original data and evaluate it.
dtc.fit(X_train, y_train)
y_pred_original = dtc.predict(X_test)
f1_original = f1_score(y_test, y_pred_original)
precision_original = precision_score(y_test, y_pred_original)
recall_original = recall_score(y_test, y_pred_original)
# Calculate the relevant indicators of the confusion matrix
cm_pca_2 = confusion_matrix(y_test, y_pred_pca_2)
tp_pca_2 = cm_pca_2[1,1]
fp_pca_2 = cm_pca_2[0,1]
fpr_pca_2 = fp_pca_2/np.sum(cm_pca_2[0,:]) if np.sum(cm_pca_2[0,:])!=0 else 0
tpr_pca_2 = tp_pca_2/np.sum(cm_pca_2[1,:]) if np.sum(cm_pca_2[1,:])!=0 else 0

print("Model using only the first principal component:")
print(f"F1 score: {f1_pca_1}, Precision: {precision_pca_1}, Recall: {recall_pca_1}")
print("Model using the first and second principal components:")
print(f"F1 score: {f1_pca_2}, Precision: {precision_pca_2}, Recall: {recall_pca_2}")
print("Model using the original continuous data:")
print(f"F1 score: {f1_original}, Precision: {precision_original}, Recall: {recall_original}")
print("Confusion matrix metrics of the model using the first and second principal components:")
print(f"TP: {tp_pca_2}, FP: {fp_pca_2}, FPR: {fpr_pca_2}, TPR: {tpr_pca_2}")
if f1_original > f1_pca_1 and f1_original > f1_pca_2:
    print("In this case, using continuous data is beneficial for the model because the F1 score of the original data model is higher, possibly retaining more effective feature information.")
elif f1_pca_1 > f1_original and f1_pca_1 > f1_pca_2:
    print("In this case, PCA dimensionality reduction using only the first principal component is beneficial for the model, possibly removing noise and correlations.")
elif f1_pca_2 > f1_original and f1_pca_2 > f1_pca_1:
    print("In this case, PCA dimensionality reduction using the first and second principal components is beneficial for the model, balancing information retention and dimensionality reduction effects.")
else:
    print("It is difficult to make a simple judgment, and further analysis of each model's indicators and data characteristics is required.")