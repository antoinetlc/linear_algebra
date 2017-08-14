# coding: utf8
import os
import numpy as np

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

def main():
    A = np.array([[1,1,0],[1,2,1],[-2,-3,1]])
    print(A)
    Q,R = QRDecomposition(A)
    print(Q)
    print(R)

if __name__ == "__main__":
    main()