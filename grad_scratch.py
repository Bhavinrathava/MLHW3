import numpy as np
import starter3
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(predictions, targets):
    epsilon = 1e-15
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    N = predictions.shape[0]
    ce_loss = -np.sum(np.log(predictions[np.arange(N), targets] + 1e-9)) / N
    return ce_loss

def compute_gradients(X, y, weights):
    # Forward pass
    h = np.dot(X, weights[0])
    h_relu = sigmoid(h)
    y_pred = softmax(np.dot(h_relu, weights[1]))

    # Backward pass
    grad_y_pred = y_pred
    grad_y_pred[np.arange(len(y)), y] -= 1
    grad_w2 = np.dot(h_relu.T, grad_y_pred)

    grad_h_relu = np.dot(grad_y_pred, weights[1].T)
    grad_h = grad_h_relu * h_relu * (1 - h_relu)
    grad_w1 = np.dot(X.T, grad_h)

    return grad_w1, grad_w2

def update_weights(weights, gradients, learning_rate):
    weights[0] -= learning_rate * gradients[0]
    weights[1] -= learning_rate * gradients[1]

def classify_insurability_manual(device,preprocess=False, early_stopping=True):
    train_data = starter3.read_insurability('three_train.csv')
    valid_data = starter3.read_insurability('three_valid.csv')
    test_data = starter3.read_insurability('three_test.csv')

    if preprocess:
        starter3.preprocess_data(train_data)
        starter3.preprocess_data(valid_data)
        starter3.preprocess_data(test_data)

    # Extract features and labels
    train_features = np.array([row[1] for row in train_data])
    train_labels = np.array([row[0][0] for row in train_data])

    valid_features = np.array([row[1] for row in valid_data])
    valid_labels = np.array([row[0][0] for row in valid_data])

    test_features = np.array([row[1] for row in test_data])
    test_labels = np.array([row[0][0] for row in test_data])

    input_size = train_features.shape[1]
    hidden_size = 2
    output_size = 3

    # Initialize weights
    weights = [np.random.randn(input_size, hidden_size), np.random.randn(hidden_size, output_size)]

    epochs = 50
    learning_rate = 0.05

    for epoch in range(epochs):
        for i in range(len(train_features)):
            X = train_features[i:i + 1]
            y = train_labels[i:i + 1]

            gradients = compute_gradients(X, y, weights)
            update_weights(weights, gradients, learning_rate)

    # Evaluate on the test set
    test_predictions = softmax(np.dot(sigmoid(np.dot(test_features, weights[0])), weights[1]))
    test_loss = cross_entropy_loss(test_predictions, test_labels)

    # Calculate accuracy
    predicted_labels = np.argmax(test_predictions, axis=1)
    accuracy = np.mean(predicted_labels == test_labels) * 100.0

    print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')