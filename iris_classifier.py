from sklearn.datasets import load_iris
# Thay thế thuật toán RandomForest bằng LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle as pickle

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

print(X)
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape)
print(y_train.shape)
# Train a RandomForestClassifier
# Thay thế thuật toán RandomForest bằng LogisticRegression
clf = LogisticRegression()
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))

print("Saving model to pickle file.")
pickle.dump(clf, open("iris_model.pkl", 'wb'))
