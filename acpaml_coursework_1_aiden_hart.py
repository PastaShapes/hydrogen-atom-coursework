import numpy as np
import matplotlib.pyplot as plt

num_points = 1000

def potential(r, l=0):
    """ 
    Function to compute the H atom potential

    Parameters
    ----------
    r : np.ndarray
        1d array of discrete radial positions
    l : int
        Angular momentum quantum number, l
    
    Returns
    ----------
    V : np.ndarray
        1D array of potential values V(r)
    """

    V = -2/r + l*(l+1)/r**2

    return V

def hamiltonian(r, V):
    """
    Function to compute the Hamiltonian

    Parameters
    ----------
    r: np.ndarray
        1D array of discrete positions
    V: np.ndarray
        1D array of potential values V(r)

    Returns
    ----------
    H : np.ndarray
        2D array (matrix) representing the Hamiltonian H
    """

    # First we create an empty matrix of zeros for the kinetic energy matrix
    K = np.zeros((len(r), len(r)))

    # The spacing between the discrete radial positions 
    dr = r[1] - r[0] 

    """ deprecated method for generating the tridiagonal matrix for the kinetic energy operator
    for i in range(len(r)):
        if i > 0:
            K[i, i-1] = -1/(dr**2)
        K[i, i] = 2/(dr**2)
        if i < len(r)-1:
            K[i, i+1] = -1/(dr**2)
    """

    # I used ChatGPT to help generate the tridiagonal matrix for the kinetic energy operator, which is a faster and more lightweight version of the commented out code above.
    K = np.diag(np.full(len(r), 2 / dr**2))
    K += np.diag(np.full(len(r)-1, -1 / dr**2), k=1)
    K += np.diag(np.full(len(r)-1, -1 / dr**2), k=-1)

    # Turn our array V into a diagonal matrix 
    V = np.diag(V)

    # We define our Hamiltonian as the sum of the kinetic and potential energy matrices
    H = K + V

    return H

def eigenvalues():
    """
    Function to compute the eigenvalues of the H atom
    Returns an array of eigenvalues, corresponding to at least the first 10 bound states, in ascending order, starting with the ground state.

    Returns 
    ----------
    eigenvalues : np.ndarray
        1D array of eigenvalues of the H atom
    """

    # We use the numpy linear algebra module to compute the eigenvalues of the Hamiltonian
    eigenvals, _ = np.linalg.eig(H)

    # We sort the eigenvalues in ascending order
    eigenvals = np.sort(eigenvals)

    return eigenvals

def validate():
    """
    Function to validate the code
    """

    # To validate our potential energy function code, we plot the potential energy array as a function of r to determine whether it is the shape that we expect for the H atom potential.
    plt.plot(r, V)
    plt.title("Potential Energy Function")
    plt.xlabel("r")
    plt.ylabel("V(r)")
    plt.show()

    # To validate our Hamiltonian code, we plot the kinetic energy matrix and the potential energy matrix
    plt.imshow(H[:16, :16], cmap='hot', interpolation='nearest')
    plt.title("Hamiltonian Matrix")
    plt.show()

if __name__ == "__main__":
    r = np.linspace(0.01, 50, num_points)
    V = potential(r)
    H = hamiltonian(r, V)
    eigenvals = eigenvalues()
    validate()