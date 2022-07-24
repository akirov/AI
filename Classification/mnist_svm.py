from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import cross_val_score, cross_val_predict, learning_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
import os
import joblib
import cv2
import sys
sys.path.append('../Vision/OpenCV/')
import cv_draw as cvd


def test_model(model):
    while True:
        # Draw with the mouse
        img = cvd.draw(384, 384, 'Enter to recognize. Esc to close')
        if img is None: break
        newimg = cv2.resize(img, (28, 28), cv2.INTER_AREA)
        plt.imshow(newimg, cmap='gray') # , cmap="Greys"
        plt.show()
        # cv2.imwrite("digit.png", newimg)
        # Make a prediction
        newimg = newimg.reshape(784)
        predicted = model.predict([newimg])
        print("predicted = ", predicted)


# Metrics.
# accuracy: correctly predicted labels part of all = (TP+TN)/(TP+FP+TN+FN)
# Hands-On Machine Learning with Scikit-Learn, 2nd ed - A.Geron, 2019:
# "... accuracy is generally not the preferred performance measure for classifiers,
# especially when you are dealing with skewed data-sets (i.e., when some
# classes are much more frequent than others)." In such case we can always
# predict true or false with high accuracy (which is not a good classifier).
# Better metrics:
# - precision: what part of the predicted positives are true positives = TP/(TP+FP).
# - sensitivity (recall): what part of all positives we have detected = TP/(TP+FN).
# - F1 score = 2*P*R/(P+R)
# "... in some contexts you mostly care about precision, and in other contexts
# you really care about recall. For example, if you trained a classifier to detect videos
# that are safe for kids, you would probably prefer a classifier that rejects many
# good videos (low recall) but keeps only safe ones (high precision), rather than a classifier
# that has a much higher recall but lets a few really bad videos show up in your
# product (in such cases, you may even want to add a human pipeline to check the classifier’s
# video selection). On the other hand, suppose you train a classifier to detect
# shoplifters in surveillance images: it is probably fine if your classifier has only 30%
# precision as long as it has 99% recall (sure, the security guards will get a few false
# alerts, but almost all shoplifters will get caught)."
#
# "When evaluating different settings ("hyperparameters") for estimators, such as the C
# setting that must be manually set for an SVM, there is still a risk of overfitting on
# the test set because the parameters can be tweaked until the estimator performs optimally.
# This way, knowledge about the test set can "leak" into the model and evaluation metrics
# no longer report on generalization performance. To solve this problem, yet another part
# of the dataset can be held out as a so-called "validation set": training proceeds on the
# training set, after which evaluation is done on the validation set, and when the
# experiment seems to be successful, final evaluation can be done on the test set.
# However, by partitioning the available data into three sets, we drastically reduce the
# number of samples which can be used for learning the model, and the results can depend
# on a particular random choice for the pair of (train, validation) sets.
# A solution to this problem is a procedure called cross-validation (CV for short). A test
# set should still be held out for final evaluation, but the validation set is no longer
# needed when doing CV. In the basic approach, called k-fold CV, the training set is split
# into k smaller sets.
# The following procedure is followed for each of the k "folds":
# - A model is trained using k-1 of the folds as training data;
# - The resulting model is validated on the remaining part of the data (i.e.,
#  it is used as a test set to compute a performance measure such as accuracy).
# The performance measure reported by k-fold cross-validation is then the average of
# the values computed in the loop."

def evaluate_model(model, X_train, y_train, X_test, y_test):
    y_test_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_test_pred)
    print("accuracy = ", accuracy)
    # "cross_val_score() clones the estimator before fitting the fold training data to it.
    # cross_val_score() will give you output an array of scores which you can analyse to
    # know how the estimator performs for different folds of the data."
    #cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy")
    #print("cv_scores = ", cv_scores, ", mean CV score = ", cv_scores.mean())

    cm = confusion_matrix(y_test, y_test_pred)
    print("Confusion Matrix (test):\n", cm)
    # Generate cross-validated estimates for each input data point
    #y_train_pred_cv = cross_val_predict(model, X_train, y_train, cv=3)
    #cm_cv = confusion_matrix(y_train, y_train_pred_cv)
    #print("Confusion Matrix (train CV):\n", cm_cv)

    print("Calculating learning curves...")
    train_sz, train_lc, val_lc = learning_curve(model, X_train, y_train, cv=3, scoring="accuracy",
                                                train_sizes=np.linspace(0.2, 1.0, 10))
    axes = plt.gca()
    axes.set_title('Accuracy Learning Curves')
    axes.text(0.7, 0.95, 'Train', transform=axes.transAxes, color="r", fontsize=10)
    plt.plot(train_sz, np.mean(train_lc, axis=1), 'o-', color="r")
    axes.text(0.7, 0.90, 'CV (test)', transform=axes.transAxes, color="g", fontsize=10)
    plt.plot(train_sz, np.mean(val_lc, axis=1), 'o-', color="g")
    plt.show()


mnist = fetch_openml('mnist_784', data_home='../data/')  #, version=1, return_X_y=True, as_frame=False
                                                         # gets data from ../data/ or tries to download it
X_full = mnist.data
print("X = mnist.data shape : ", X_full.shape)
y_full = mnist.target.astype(np.uint8)
print("y = mnist.target shape : ", y_full.shape)

# Plot 5 random MNIST images with labels
fig, axs = plt.subplots(1, 5)
fig.suptitle('5 MNIST images with index:label')
for i in range(5):
  idx = np.random.randint(0,X_full.shape[0])
  img = (np.reshape(X_full.to_numpy()[idx], (28, 28))).astype(np.uint8)
  axs[i].set_title('{0}:{1}'. format(idx, y_full[idx]))
  axs[i].imshow(img, interpolation='nearest', cmap='gray')  # cmap='binary'
  axs[i].axis('off')
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2)


# "One way to create a system that can classify the digit images into 10 classes
# (from 0 to 9) is to train 10 binary classifiers, one for each digit. This is called
# the one-versus-the-rest (OvR) strategy (also called # one-versus-all)."
# "Another strategy is to train a binary classifier for every pair of digits: one to
# distinguish 0s and 1s, another to distinguish 0s and 2s, another for 1s and 2s, and
# so on. This is called the one-versus-one (OvO) strategy. If there are N classes, you
# need to train N × (N – 1) / 2 classifiers."
# "Some algorithms (such as Support Vector Machine classifiers) scale poorly with the
# size of the training set. For these algorithms OvO is preferred because it is faster to
# train many classifiers on small training sets than to train few classifiers on large
# training sets. For most binary classification algorithms, however, OvR is preferred."
# "Scikit-Learn detects when you try to use a binary classification algorithm for a
# multiclass classification task, and it automatically runs OvRest or OvOne, depending
# on the algorithm."

# SVM with linear kernel
linear_svm = LinearSVC()  # C=1, multi_class='ovr'

if os.path.isfile(os.path.abspath("mnist_svm_linear.pkl")):
    print("Loading linear_svm model...")
    linear_svm = joblib.load("mnist_svm_linear.pkl")
else:
    print("Training linear_svm model...")
    linear_svm.fit(X_train, y_train)
    joblib.dump(linear_svm, "mnist_svm_linear.pkl")

print("Testing linear_svm...")
test_model(linear_svm)

print("Evaluating linear_svm...")
evaluate_model(linear_svm, X_train, y_train, X_test, y_test)

# "If your SVM model is overfitting, you can try regularizing it by
# reducing C. This increases the margin. Thus we will have more margin
# violations, but will generalize better."


# SVM with RBF kernel


# SVM with a polynomial kernel. Do we need StandardScaler()?


# PCA + linear SVM?


# Hog features with linear SVM?
