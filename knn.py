import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from numpy import mean

# Loading handwritten digits dataset
digits = load_digits()

X = digits.data
y = digits.target

# Training KNN Model and testing it with specific K values by getting its predictions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
knn = OneVsRestClassifier(KNeighborsClassifier(n_neighbors = 3))
knn.fit(X_train,y_train)
predictions = knn.predict(X_test)

# Model Cross Validation by KFold method
cv = KFold(n_splits=10, random_state=1, shuffle=True)
scores = cross_val_score(knn, X, y, scoring= 'accuracy', cv=cv, n_jobs=-1)
print('\nCross Validation Result: ')
print('Mean Accuracy: %.3f' % mean(scores))

# Classification Report
print('\nClassification Report:\n')
accuracy_score(y_test,predictions)
print(classification_report(y_test, predictions))

# Plotting Confusion Matrix using seaborn Heatmap
cm = confusion_matrix(y_test,predictions)
plt.figure(figsize=(9,9))
sns.heatmap(cm,annot=True, fmt='.3f', linewidths=.5, square=True,cmap='Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(accuracy_score(y_test,predictions)*100)+'%'
plt.title(all_sample_title,size=15)
plt.show()