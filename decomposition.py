# coding: utf8
import os
import numpy as np
import matplotlib.pyplot as plt

def PCACovariance():
    """
        Computes the PCA of the data matrix X.
        N observations of dimensionality d.
        Dimension along the rows and observations along the columns.
        """
    
    N = 1000
    
    mean = [0,1]
    sigma = [[2,0.0],[0.0,1.0]]
    X = np.random.multivariate_normal(mean, sigma, N).T
    
    #Remove the mean
    one = np.ones((N,1)) #Column vector full of ones
    mu = 1.0/N*X.dot(one)
    XCentered = X-mu.dot(one.T)
    
    #Compute the covariance matrix and its eigen decomposition
    #U contains the eigenvectors of the covariance matrix in its rows
    eigValues, U = np.linalg.eig(XCentered.dot(XCentered.T))
    Lambda = np.diag(np.array(eigValues))
   
    #Plot
    fig, ax = plt.subplots()
    ax.plot(XCentered[0,:], XCentered[1,:], '.')
    
    #Plot axis scaled by sqrt(eigenvalues)
    origin = [0], [0] #Origin
    plt.quiver(*origin, np.sqrt(Lambda[0,0])*U[0,:], np.sqrt(Lambda[1,1])*U[1,:], color=['r','b','g'], scale=21)
    plt.axis('equal')
    plt.show()

def PCA():
    """
        Computes the PCA of the data matrix X.
        N observations of dimensionality d.
        Dimension along the rows and observations along the columns.
    """
    
    """N = 1000
    
    mean = [0,1]
    sigma = [[1,-1],[-1,10]]
    X = np.random.multivariate_normal(mean, sigma, N).T
    
    #Remove the mean
    print(np.shape(X))

    #PCA
    #The number of dimensions is too low here compared to the observations
    #Hence computing the eigen decomposition of the covariance matrix XX^T would be faster
    #However in real world problems, d>>N and computing he eigen decomposition of the covariance matrix is very expensive
    #Therefore the eigendecomposition of X^TX is computed and then the eigenvectors of XX^T are recovered.
    #Eigen values of XX^T
    eigValues, V = np.linalg.eig(X.dot(X.T))
    Lambda = np.diag(eigValues)
    print(eigValues)
    #The eigenvectors of XX^T are related by U = XVeigValues^-1/2
    U = X.dot(V).dot(np.sqrt(np.linalg.inverse(Lambda)))
    print(U.shape)
    
    fig, ax = plt.subplots()
    ax.plot(X[0,:], X[1,:], '.')
    plt.axis('equal')
    plt.show()"""

def QRDecomposition(A):
    """
        Outputs the QR decomposition given an input basis.
    """
    
    if (A.shape[0] != A.shape[1]) or np.abs(np.linalg.det(A))<0.0001:
        print("Input matrix is not a basis")
    
    else:
        Q = np.zeros(A.shape)
        R = np.zeros(A.shape)
        
        #First vector
        Q[:,0] = A[:,0]/np.linalg.norm(A[:,0])
        R[0,0] = np.linalg.norm(A[:,0])
        
        #Q is the matrix of the new orthonormal vectors
        #R contains the scaling factors that are applied in each linear combination in the Gram Schmidt process
        for j in range(1, A.shape[1]):
            Q[:,j] = A[:,j]
            
            for k in range(0, j):
                Q[:,j] -= (Q[:,k].T.dot(A[:,j]))*Q[:,k]
                R[k,j] = Q[:,k].T.dot(A[:,j])
        
            #Normalize
            Q[:,j] = Q[:,j]/np.linalg.norm(Q[:,j])
            R[j,j] = Q[:,j].T.dot(A[:,j])
    
        return Q, R

def QRAlgorithm(A, iterations):
    """
        Computes the eigenvalues of matrix A using the QR algorithm.
    """
    
    currentA = A
    
    for i in range(0, iterations):
        Q, R = np.linalg.qr(currentA)
        currentA = Q.T.dot(currentA).dot(Q)

    return currentA

def comparisonEigenvalues(iterations):
    """
        Computes the eigenvalues of matrix A using the QR algorithm and numpy and compares the difference.
    """
    A = np.random.rand(100,100)
    B = A.T.dot(A)
    B = (B+B.T)/2.0

    #QR algorithm
    eigenvaluesQR = np.diagonal(QRAlgorithm(B, iterations))
    eigenvaluesQR = np.sort(eigenvaluesQR)
    eigenvaluesQR = eigenvaluesQR[::-1] #Reverse the array to get larger value first
    #print(eigenvaluesQR[::-1])

    #numpy
    eigenvaluesNumpy, eigenVectors = np.linalg.eig(B)
    #print(eigenvaluesNumpy)

    #Computes difference
    difference = np.sqrt(np.sum(np.square(eigenvaluesNumpy-eigenvaluesQR)))
    print("Difference between eigenvalues : %f" % difference)

def QRDecompositionExample():
    """
        Example of how to use the QR decomposition.
    """
    A = np.array([[1,1,0],[1,2,1],[-2,-3,1]])
    print(A)
    Q,R = QRDecomposition(A)
    print(Q)
    print(R)

def main():
    PCACovariance()

if __name__ == "__main__":
    main()