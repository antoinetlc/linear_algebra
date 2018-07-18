# coding: utf8
import os
import numpy as np
import matplotlib.pyplot as plt

def gaussianPDF_1D(x, mu, sigma):
    """
        One dimensional probability density function of the Gaussian
    """
    var = sigma*sigma

    return 1.0/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-mu)**2/(2*var))

def metropolis_hastings():
    """
        Example of the metropolis hasting algorithm for a 1D gaussian (Markov Chain Monte Carlo).
        Assumption : we would like to sample from a normal distribution N(0,1)
        Proposal distribution q(x(t+1)/x(t)) = N(x(t), sigma). sigma fixed.
    """
    #Number of samples
    N = 100000
    
    #Known sigma from the proposal distribution
    sigma = 10.0
    
    #List of samples : starts from 10.0 for instance
    x = [1.0]
   
    #Main loop : sample N times from the distribution
    i=1
    while i<N:
        #Sample from proposal distribution
        x_potential = np.random. normal(x[i-1], sigma)

        #Computes the Metropolis Hastings ratio
        ratio = gaussianPDF_1D(x[i-1], x_potential, sigma)*np.exp(-x_potential*x_potential/2.0)/(gaussianPDF_1D(x_potential, x[i-1], sigma)*np.exp(-x[i-1]*x[i-1]/2.0))

        #Accept or reject the sample ?
        if(ratio >= np.random.rand()):
            x.extend([x_potential])
            i = i+1

        #Print current step
        print('%d %.4f' % (i, ratio))
        

    #Plot the histogram
    count, bins, ignored = plt.hist(x, 30, normed=True)
    plt.plot(bins, gaussianPDF_1D(bins, 0, 1), linewidth=2, color='r')
    plt.show()

def main():
    metropolis_hastings()

if __name__ == "__main__":
    main()