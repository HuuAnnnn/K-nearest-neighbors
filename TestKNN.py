from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from KNN import KNN

iris = datasets.load_iris()
iris_x = iris.data
iris_y = iris.target
x_train, x_test, y_train, y_test = train_test_split(
    iris_x, iris_y, test_size=50)


knn = KNN(k_neighbors=1, p=2)
knn.load_data(x_train, y_train)

y_pred = knn.predict(x_test)
print(f"Predict datapoint: {y_pred}")
print(f"Ground truth: {list(y_test)}")

print(accuracy_score(y_pred, y_test)*100)
