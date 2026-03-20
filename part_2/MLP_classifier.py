import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def main():
    # Load the Iris flower dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Split the dataset into 80% training and 20% testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("=== Task 2(2)(a): Simple MLP Classifier ===")
    # Note: Scikit-learn automatically determines the number of output neurons 
    # based on the number of classes in the target variable 'y' (which is 3 for Iris).
    mlp_simple = MLPClassifier(
        hidden_layer_sizes=(3,), # One hidden layer with 3 neurons
        activation='relu',       # Activation function
        solver='adam',           # Optimizer
        learning_rate_init=0.01, # Learning rate
        max_iter=1000,           # Number of epochs
        batch_size=32,           # Batch size
        random_state=42          # Seed for initial weight/bias vector reproducibility
    )

    # Train the model
    mlp_simple.fit(X_train, y_train)

    # Compute training and testing accuracy
    y_train_pred = mlp_simple.predict(X_train)
    y_test_pred = mlp_simple.predict(X_test)
    train_acc_simple = accuracy_score(y_train, y_train_pred) * 100
    test_acc_simple = accuracy_score(y_test, y_test_pred) * 100

    print(f"Training Accuracy: {train_acc_simple:.2f}%")
    print(f"Testing Accuracy: {test_acc_simple:.2f}%")
    print(f"Epochs run until convergence: {mlp_simple.n_iter_}")
    
    # ---------------------------------------------------------
    
    print("\n=== Task 2(2)(b): Increased Complexity MLP ===")
    # Increase the network complexity by adding more layers and neurons
    mlp_complex = MLPClassifier(
        hidden_layer_sizes=(10, 10), # Two hidden layers with 10 neurons each
        activation='relu',       
        solver='adam',           
        learning_rate_init=0.01, 
        max_iter=1000,           
        batch_size=32,           
        random_state=42          
    )

    # Train the more complex model
    mlp_complex.fit(X_train, y_train)

    # Compute training and testing accuracy
    y_train_pred_complex = mlp_complex.predict(X_train)
    y_test_pred_complex = mlp_complex.predict(X_test)
    train_acc_complex = accuracy_score(y_train, y_train_pred_complex) * 100
    test_acc_complex = accuracy_score(y_test, y_test_pred_complex) * 100

    print(f"Training Accuracy: {train_acc_complex:.2f}%")
    print(f"Testing Accuracy: {test_acc_complex:.2f}%")
    print(f"Epochs run until convergence: {mlp_complex.n_iter_}")

if __name__ == "__main__":
    main()