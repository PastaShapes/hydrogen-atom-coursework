import numpy as np
import matplotlib.pyplot as plt

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

    # Define the spacing between the discrete radial positions 
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

    num_points = 2500
    r = np.linspace(0, 250, num_points)[1:]
    
    V = potential(r, l=0)
    H = hamiltonian(r, V)

    # We use the numpy linear algebra module to compute the eigenvalues of the Hamiltonian, also .eigh() auto-sorts to ascending order
    eigenvals, _ = np.linalg.eigh(H)

    return eigenvals

def validate():
    """
    Function to validate the code
    Plots a few visualisations so we can validate the code as well as comparing results to expected values
    """

    # no. of points - we choose 2500s to have a sufficient amount of points to mitigate error but small enough to maintain reasonable performance
    num_points = 2500
    # lower bound - we start the linspace at 0 but slice [1:] to avoid division by zero
    # upper bound - we choose 250 as the upper bound to ensure we capture the first 10 bound states without box confinement errors

    r = np.linspace(0, 250, num_points)[1:]
    V = potential(r)
    H = hamiltonian(r, V)
    eigenvals = eigenvalues()
    
    # Plot the potential energy array as a function of r to determine whether it is the shape that we expect 
    plt.subplot(1, 2, 1)
    plt.plot(r, V)
    plt.title("Potential Energy Function")
    plt.xlabel("r")
    plt.ylabel("V(r)")
    plt.ylim(-10, 2) # Limit y-axis to see the potential well clearly

    # Kinetic energy matrix and the potential energy matrix
    plt.subplot(1, 2, 2)
    plt.imshow(H[:30, :30], cmap='hot', interpolation='nearest')
    plt.title("Hamiltonian Matrix (Top Left)")
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()

    # Print the eigenvalues of the H atom we calculated compared with the analytical solution
    print("Analytical energy eigenvalues of the H atom (in ascending order):")
    analytical_eigenvals = -1/(np.arange(1, 11)**2)  # we obtain this from our energy levels defined as En = -1/(2n^2) with dimensionless units
    print(analytical_eigenvals)
    print("Energy eigenvalues of the H atom (in ascending order):")
    print(eigenvals[:10])  # print the first 10 eigenvalues


    # Comparison with simple harmonic oscillator potential
    x = np.linspace(-10, 10, num_points)
    V_SHO = x**2

    H_SHO = hamiltonian(x, V_SHO)
    eigenvals_SHO, _ = np.linalg.eig(H_SHO)
    eigenvals_SHO = np.sort(eigenvals_SHO)

    print("\nAnalytical energy eigenvalues of the Simple Harmonic Oscillator (in ascending order):")
    analytical_eigenvals_SHO = 2 * np.arange(0, 10) + 1 # Analytical solution for the first 10 energy levels of the SHO
    print(analytical_eigenvals_SHO)

    print("\nEnergy eigenvalues of the Simple Harmonic Oscillator (in ascending order):")
    print(eigenvals_SHO[:10])  # print the first 10 eigenvalues

if __name__ == "__main__":
    validate()