# coding: utf8
import os
import numpy as np

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
    
    comparisonEigenvalues(1)
    comparisonEigenvalues(10)
    comparisonEigenvalues(100)
    comparisonEigenvalues(1000)
    comparisonEigenvalues(10000)

if __name__ == "__main__":
    main()