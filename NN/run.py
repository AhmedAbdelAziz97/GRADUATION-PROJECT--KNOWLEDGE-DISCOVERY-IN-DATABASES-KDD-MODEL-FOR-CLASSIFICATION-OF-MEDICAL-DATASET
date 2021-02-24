import numpy as np
from NeuralNetworkClass import NeuralNetwork
import util as utils
import pandas as pd

def main():
    # ===================================
    # Settings
    # ===================================
    filename = "Datasets\data-after-transform no headings.csv"
    n_hidden_nodes = [5]  # nodes in hidden layers i.e. [n_nodes_1, n_nodes_2, ...]
    l_rate = 0.6  # learning rate
    n_epochs = 800  # number of training epochs
    n_folds = 4  # number of folds for cross-validation

    print("Neural network model:\n n_hidden_nodes = {}".format(n_hidden_nodes))
    print(" l_rate = {}".format(l_rate))
    print(" n_epochs = {}".format(n_epochs))
    print(" n_folds = {}".format(n_folds))

    # ===================================
    # Read data (X,y) and normalize X
    # ===================================
    print("\nReading '{}'...".format(filename))
    utils.reader
    X, y = utils.read_csv(filename)  # read as matrix of floats and int
    utils.normalize(X)  # normalize
    N, d = X.shape  # extract shape of X
    n_classes = len(np.unique(y))

    print(" X.shape = {}".format(X.shape))
    print(" y.shape = {}".format(y.shape))
    print(" n_classes = {}".format(n_classes))

    # ===================================
    # Create cross-validation folds
    # These are a list of a list of indices for each fold
    # ===================================
    idx_all = np.arange(0, N)
    idx_folds = utils.crossval_folds(N, n_folds, seed=1)

    # ===================================
    # Train and evaluate the model on each fold
    # ===================================
    acc_train, acc_test = list(), list()  # training/test accuracy score
    print("\nTraining and cross-validating...")
    for i, idx_test in enumerate(idx_folds):

        # Collect training and test data from folds
        idx_train = np.delete(idx_all, idx_test)
        X_train, y_train = X[idx_train], y[idx_train]
        X_test, y_test = X[idx_test], y[idx_test]

        # Build neural network classifier model and train
        model = NeuralNetwork(n_input=d, n_output=n_classes, n_hidden_nodes=n_hidden_nodes)
        model.train(X_train, y_train, l_rate=l_rate, n_epochs=n_epochs)

        # Make predictions for training and test data
        y_train_predict = model.predict(X_train)
        y_test_predict = model.predict(X_test)

        # Compute training/test accuracy score from predicted values
        acc_train.append(100*np.sum(y_train==y_train_predict)/len(y_train))
        acc_test.append(100*np.sum(y_test==y_test_predict)/len(y_test))

        # Print cross-validation result
        print(" Fold {}/{}: train acc = {:.2f}%, test acc = {:.2f}% (n_train = {}, n_test = {})".format(i+1, n_folds, acc_train[-1], acc_test[-1], len(X_train), len(X_test)))

    # ===================================
    # Print results
    # ===================================
    print("\nAvg train acc = {:.2f}%".format(sum(acc_train)/float(len(acc_train))))
    print("Avg test acc = {:.2f}%".format(sum(acc_test)/float(len(acc_test))))


# Driver
if __name__ == "__main__":
    main()
