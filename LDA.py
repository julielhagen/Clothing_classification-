# Imports
# Numoy
import numpy as np

# Scipy
from scipy.linalg import inv
from scipy.linalg import eig

class LDA():
    def __init__(self, n_components):
        self.n_components = n_components
        self.linear_discriminants = None

    def fit(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)
        
        means = np.mean(X, axis=0)
        SW = np.zeros((n_features, n_features))
        SB = np.zeros((n_features, n_features))

        for c in class_labels:
            # Find within class scatter matrix S_W:
            X_c = X[y==c]
            #Find mean for ach class
            m_i = np.mean(X_c, axis=0)

            # Sum of SW_c using formula for SW
            SW += (X_c -m_i).T.dot((X_c-m_i))

            # Find between class scatter matrix
            n_c = X_c.shape[0]
            mean_diff = (m_i - means).reshape(n_features,1)

            # Sum of SB
            SB += n_c * (mean_diff).dot(mean_diff.T)

        #Find the eigenvector for S_W^-1SW:
                   
        # calculate the inverse of SW
        SW_inv = inv(SW)

        # Calculate SW^-1SB
        A = SW_inv.dot(SB)

        # Find the eigenvectors and eigenvalues of A
        eigenvalues, eigenvectors = eig(A)
        eigenvectors = eigenvectors.T

        # Get indexes of maximum eigenvalues
        idx = np.argsort(eigenvalues.real)[::-1]

        # Get sorted eigenvalues and eigenvectors
        eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[idx].real

        # Save the n first eigenvectors
        self.linear_discriminants = eigenvectors[0:self.n_components]
    
    def transform(self, X):
        return np.dot(X, self.linear_discriminants.T)
    



