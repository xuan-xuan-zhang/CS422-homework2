import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def load_preprocess_data():
    try:
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['target'] = iris.target

        X = df.drop('target', axis=1)
        y = df['target']
        return X, y
    except Exception as e:
        print(f"Error in data loading or preprocessing: {e}")
        raise
#Training models and evaluation functions
def train_and_evaluate_models(x, y):
    try:
        #Divide the dataset into a training set and a test set.
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        best_recall_depth=0
        highest_recall=0
        worst_precision_depth=0
        lowest_precision=1
        best_f1_depth=0
        best_f1=0
        #Traverse the depth of the decision tree
        for depth in range(1, 11):
            tree = DecisionTreeClassifier(min_samples_leaf=2, min_samples_split=5, max_depth=depth, random_state=42)
            tree.fit(X_train, y_train)
            y_pred = tree.predict(X_test)
            recall = recall_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            if recall > highest_recall:
                highest_recall = recall
                best_recall_depth = depth
            if precision < lowest_precision:
                lowest_precision = precision
                worst_precision_depth = depth
            if f1 > best_f1:
                best_f1 = f1
                best_f1_depth = depth

        highest_recall = round(highest_recall, 4)
        lowest_precision = round(lowest_precision, 4)
        best_f1 = round(best_f1, 4)

        return best_recall_depth, worst_precision_depth, best_f1_depth, highest_recall, lowest_precision, best_f1

    except Exception as e:
        print(f"Error on model training or evaluation: {e}")
        raise

X, y = load_preprocess_data()
best_recall_depth, worst_precision_depth, best_f1_depth, highest_recall, lowest_precision, best_f1 = train_and_evaluate_models(X, y)
print(f"Best Recall Depth: {best_recall_depth}, Highest Recall: {highest_recall}")
print(f"Worst Precision Depth: {worst_precision_depth}, Lowest Precision: {lowest_precision}")
print(f"Best F1 Depth: {best_f1_depth}, Best F1: {best_f1}")