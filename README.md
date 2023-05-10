# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python for finding linear regression.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Predict the values of array.
5. Calculate the accuracy, confusion and classification report b   importing     the required modules from sklearn.
6. Obtain the graph.

## Program:
```python
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Chandhuru S
RegisterNumber: 212220220007 
*/




import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)



def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()


plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
np.mean(predict(res.x,X)==y)

```

## Output:

![c1](https://github.com/ChandhuruS/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/123381860/57886e00-92ee-4234-84c8-43231820e720)
![c2](https://github.com/ChandhuruS/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/123381860/846b3fb3-c19c-4a90-882c-21065bfd9ce3)
![c3](https://github.com/ChandhuruS/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/123381860/a26f4fe5-e56b-41f3-92e1-31bdb9c9d5b8)

![c4](https://github.com/ChandhuruS/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/123381860/32b2e970-30ea-4d86-8d40-d927c3befe0f)

![c5](https://github.com/ChandhuruS/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/123381860/1bc11204-efc2-48f2-a71d-eedc3fa949fd)
![c6](https://github.com/ChandhuruS/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/123381860/c3a17775-f125-4650-bf9c-e2d9eb9eac7c)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

