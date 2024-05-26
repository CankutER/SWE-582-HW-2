import numpy as np
from random import choice
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def nonlinear_svm(X_train,y_train,X_test,y_test):
    svm = SVC(kernel='poly',max_iter=10000)

    # Define the hyperparameters
    param_grid = {
        'C': [0.01,0.1,0.5], # Values for C to control the amount of slack
        'degree':[2,3], #dimensionality
        'coef0':[0.01,0.1,0.5]  #fixed coef in polynomial kernel function
    }

    # Find the best hyperparameters with 5-fold cross-validation
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_svm = grid_search.best_estimator_

    # Train and evaluate the best model
    best_svm.fit(X_train, y_train)
    y_train_pred = best_svm.predict(X_train)
    y_test_pred = best_svm.predict(X_test)

    # Calculate accuracy
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    cv_results = grid_search.cv_results_
    print("Mean Test Score in CV for Each Parameter Combination:")
    for mean_score, std_score, params in zip(cv_results['mean_test_score'], cv_results['std_test_score'], cv_results['params']):
        print(f"Mean Test Score: {mean_score:.4f} (Std: {std_score:.4f}), Parameters: {params}")

    print(f"Best hyperparameters: {grid_search.best_params_}")
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    support_vectors_indices = best_svm.support_
    support_vectors = X_train[support_vectors_indices]
    n_support_vectors_to_inspect = 10
    plot_images(support_vectors[:n_support_vectors_to_inspect], y_train[support_vectors_indices][:n_support_vectors_to_inspect], 'Support Vectors', 2, 5)
    non_support_vectors_indices = np.setdiff1d(np.arange(X_train.shape[0]), support_vectors_indices)
    non_support_vectors = X_train[non_support_vectors_indices]
    plot_images(non_support_vectors[:n_support_vectors_to_inspect], y_train[non_support_vectors_indices][:n_support_vectors_to_inspect], 'Non-Support Vectors', 2, 5)

def linear_svm(X_train,y_train,X_test,y_test):
    svm = LinearSVC(dual=False,max_iter=2000)

    # Define the hyperparameters
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100]  # Values for C to control the amount of slack
    }

    # Find the best hyperparameters with 5-fold cross-validation
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_svm = grid_search.best_estimator_

    # Train and evaluate the best model
    best_svm.fit(X_train, y_train)
    y_train_pred = best_svm.predict(X_train)
    y_test_pred = best_svm.predict(X_test)

    # Calculate accuracy
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    cv_results = grid_search.cv_results_
    print("Mean Test Score in CV for Each Parameter Combination:")
    for mean_score, std_score, params in zip(cv_results['mean_test_score'], cv_results['std_test_score'], cv_results['params']):
        print(f"Mean Test Score: {mean_score:.4f} (Std: {std_score:.4f}), Parameters: {params}")

    print(f"Best hyperparameters: {grid_search.best_params_}")
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

def plot_images(images, labels, title, n_row, n_col):
    plt.figure(figsize=(10, 10))
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        plt.title(f'Label: {labels[i]}')
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

def sample_data_without_overlap(X, y, labels, num_train_samples_per_class, num_test_samples_per_class):
    train_X = []
    train_y = []
    test_X = []
    test_y = []
    
    for label in labels:
        label_indices = np.where(y == label)[0]
        
        # Sample for training set
        train_indices = np.random.choice(label_indices, num_train_samples_per_class, replace=False)
        
        # Remaining indices after training samples are removed
        remaining_indices = np.setdiff1d(label_indices, train_indices)
        
        # Sample for test set from remaining indices
        test_indices = np.random.choice(remaining_indices, num_test_samples_per_class, replace=False)
        
        # Append the samples accordingly
        train_X.append(X.iloc[train_indices])
        train_y.append(y.iloc[train_indices])
        test_X.append(X.iloc[test_indices])
        test_y.append(y.iloc[test_indices])
    
    return (np.vstack(train_X), np.hstack(train_y)), (np.vstack(test_X), np.hstack(test_y))

    
if __name__ == '__main__':

    # Load the dataset
    mnist = fetch_openml('mnist_784', version=1,parser='auto')

    # Split into data and label
    X, y = mnist.data, mnist.target
    X = X / 255.0  # Normalize the images to [0, 1] range
    y = y.astype(int)

    #Anchor for Checkpoint to see if label indices are protected
    print(y.iloc[54])

    filter_labels = [2, 3, 8, 9]
    filtered_indices = np.isin(y, filter_labels)

    X_filtered = X[filtered_indices]
    y_filtered = y[filtered_indices]
    
    num_train_samples_per_class = 5000
    num_test_samples_per_class = 1000

    # Sample data for training and testing without overlap
    (X_train, y_train), (X_test, y_test) = sample_data_without_overlap(
    X_filtered, y_filtered, filter_labels, num_train_samples_per_class, num_test_samples_per_class)

    #Checkpoint to see if label indices are protected
    print(y.iloc[54])

    # Print shapes to verify
    print("Training data shape:", X_train.shape)
    print("Training labels shape:", y_train.shape)
    print("Testing data shape:", X_test.shape)
    print("Testing labels shape:", y_test.shape)

    # Verify the number of samples per class in training and testing sets
    unique_train_labels, train_counts = np.unique(y_train, return_counts=True)
    unique_test_labels, test_counts = np.unique(y_test, return_counts=True)
    print("Training set label counts:", dict(zip(unique_train_labels, train_counts)))
    print("Testing set label counts:", dict(zip(unique_test_labels, test_counts)))

    #linear_svm(X_train,y_train,X_test,y_test)
    nonlinear_svm(X_train,y_train,X_test,y_test)
   

