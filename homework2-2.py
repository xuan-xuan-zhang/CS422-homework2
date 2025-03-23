import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np
#Load data and conduct data preprocessing
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
col_names = ['Sample Code Number', 'Clump Thickness', 'Uniformity of Cell Size',
             'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size',
             'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
df = pd.read_csv(url, names=col_names, na_values='?')
df = df.dropna()
X = df.drop(['Sample Code Number', 'Class'], axis=1)#Extract features and labels
y = df['Class'].map(lambda x: 1 if x == 4 else 0)
#Train the decision tree model
clf = DecisionTreeClassifier(min_samples_leaf=2, min_samples_split=5, max_depth=2, random_state=42)
clf.fit(X, y)
#The function for calculating entropy
def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return -np.sum([pi * np.log2(pi) for pi in p if pi != 0])
# The function for calculating the Gini index
def gini(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1 - np.sum(p ** 2)
#The function for calculating the error of misclassification
def misclassification_error(y):
    _, counts = np.unique(y, return_counts=True)
    return 1 - np.max(counts / len(y))
#Search for the optimal splitting point
best_info_gain = 0
best_feature = None
best_threshold = None
for feature in X.columns:
    unique_values = np.sort(X[feature].unique())
    for i in range(len(unique_values) - 1):
        threshold = (unique_values[i] + unique_values[i + 1]) / 2
        left_y = y[X[feature] <= threshold]
        right_y = y[X[feature] > threshold]
        impurity_parent = entropy(y)
        impurity_left = entropy(left_y)
        impurity_right = entropy(right_y)
        info_gain = impurity_parent - (len(left_y) / len(y)) * impurity_left - (len(right_y) / len(y)) * impurity_right
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = feature
            best_threshold = threshold
#Calculate and output
left_y = y[X[best_feature] <= best_threshold]
right_y = y[X[best_feature] > best_threshold]
entropy_parent = entropy(y)
entropy_left = entropy(left_y)
entropy_right = entropy(right_y)
gini_parent = gini(y)
gini_left = gini(left_y)
gini_right = gini(right_y)
mce_parent = misclassification_error(y)
mce_left = misclassification_error(left_y)
mce_right = misclassification_error(right_y)

print(f"Entropy before the first split: {entropy_parent}")
print(f"Entropy of the left child node of the first split: {entropy_left}")
print(f"Entropy of the right child node of the first split: {entropy_right}")
print(f"Gini index before the first split: {gini_parent}")
print(f"Gini index of the left child node of the first split: {gini_left}")
print(f"Gini index of the right child node of the first split: {gini_right}")
print(f"Misclassification error before the first split: {mce_parent}")
print(f"Misclassification error of the left child node of the first split: {mce_left}")
print(f"Misclassification error of the right child node of the first split: {mce_right}")
print(f"Information gain: {best_info_gain}")
print(f"Feature selected for the first split: {best_feature}")
print(f"Value that determines the decision boundary: {best_threshold}")
