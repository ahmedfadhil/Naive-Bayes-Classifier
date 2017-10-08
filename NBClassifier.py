from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from adspy_shared_utilities import plot_class_regions_for_classifier

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

nbclf = GaussianNB.fit(X_train,y_train)
plot_class_regions_for_classifier(nbclf, X_train,y_train, X_test,y_test, 'Gaussian Naive Bayes Classifier:\ Dataset 1')


# Real data

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

nbclf = GaussianNB.fit(X_train,y_train)
print('Breast Cancer dataset')
print('Accuracy of GaussianNB classifier on training set:{:.2f}'.format(nbclf,X_train,y_train))
print('Accuracy of GaussianNB classifier on testing set:{:.2f}'.format(nbclf,X_test,y_test))
# plot_class_regions_for_classifier(nbclf, X_train,y_train, X_test,y_test, 'Gaussian Naive Bayes Classifier:\ Dataset 1')
