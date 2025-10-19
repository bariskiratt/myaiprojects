import numpy as np

def forward_prop(w2,w3,b2,b3,X):
    z2 = np.dot(w2, X) + b2
    a2 = sigmoid(z2)
    z3 = np.dot(w3, a2) + b3
    a3 = softmax(z3)

    return z2,a2,z3,a3


def compute_loss(Y, a3):

    loss = -np.sum(Y * np.log(a3))
    return loss


def backprop(X, Y, w3, z2, a2, z3, a3):

    m = X.shape[1]
    dZ3 = a3 - Y
    dW3 = (1/m) * np.dot(dZ3,a2.T)
    db3 = (1/m) * np.sum(dZ3, axis =1 , keepdims= True)
    dZ2 = np.dot(w3.T,dZ3) * (a2 * (1 - a2))
    dW2 = (1/m) * np.dot(dZ2,X.T)
    db2 = (1/m) * np.sum(dZ2, axis =1 , keepdims= True)
    return dW2, db2, dW3, db3


def update_params(W2, b2, W3, b3, dW2, db2, dW3, db3, learning_rate):
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    W3 = W3 - learning_rate * dW3
    b3 = b3 - learning_rate * db3

    return W2, b2, W3, b3

def main():

    X_list = [[0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]]
    Y_list = [[0,1], [1,0], [1,0], [0,1], [1,0], [0,1], [0,1], [1,0]]
    X_train = np.array(X_list).T
    Y_train = np.array(Y_list).T

    w2 = np.random.randn(4,3)
    w3 = np.random.randn(2,4)

    b2 = np.zeros((4,1))
    b3 = np.zeros((2,1))    

    learning_rate = 0.2
    epochs = 100000

    loss = 0

    for i in range(epochs+1):

        z2,a2,z3,a3 = forward_prop(w2,w3,b2,b3,X_train)
        loss = compute_loss(Y_train, a3)
        if(i % 10000 == 0):
            print(f"Epoch {i}, Loss: {loss}")    

        
        dW2, db2, dW3, db3 = backprop(X_train,Y_train,w3,z2,a2,z3,a3)
        w2,b2,w3,b3 = update_params(w2,b2,w3,b3,dW2,db2,dW3,db3,learning_rate)
    return w2,b2,w3,b3
 
w2,b2,w3,b3 = main()


def sigmoid(z):
    return 1/(1+np.exp(-z))
    
def softmax(Z):
    Zs = Z - np.max(Z, axis=0, keepdims=True)
    expZ = np.exp(Zs)
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def predict():
    X = [0,0,0]
    sum = 0
    for i in range(3):
        X[i] = int(input(f"Enter the {i+1}st neuron input:  "))

    X_test = np.array(X).T
    X_test = X_test.reshape(3,1)
    _, _, _, A3 = forward_prop(w2,w3,b2,b3,X_test)
    print(A3.ravel())
    k = int(np.argmax(A3, axis=0).item())
    etiket = ("odd" , "even")[k]
    print(f"Label: {etiket} by {A3[k] * 100}%")
predict()
