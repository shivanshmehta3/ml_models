import numpy as np
class PCA:
    @staticmethod
    def featureNormalize(X):
        X_norm = X
        [m,n] = X.shape

        number_of_training_sets = m

        mu = np.resize(X.mean(0),(1,n))
        mu_for_subtraction = np.ones((number_of_training_sets,1)) @ mu

        sigma = np.resize(np.std(X,0),(1,n))
        sigma_for_division = np.ones((number_of_training_sets,1)) @ sigma

        X_norm = np.divide((X - mu_for_subtraction),sigma_for_division)
        return X_norm, mu, sigma
    
    @staticmethod
    def pca(X):
        [m, n] = X.shape
        x_trans = X.T
        covariance_matrix_sigma = np.cov(x_trans)

        U,s,V = np.linalg.svd(covariance_matrix_sigma)
        S = np.array([[s[j] if i==j else 0 for j in range(n)] for i in range(m)])
        print(S)
        print(S.shape)
        return U,S
    
    @staticmethod
    def projectData(X,U,K):
        [m,n] = X.shape
        Z = np.zeros((m, K))

        U_reduced = U[:, 0:K]
        print(K)
        print(m,n)
        print(U.shape)
        print(U_reduced.shape)
        Z = X @ U_reduced

    @staticmethod
    def varianceRetained(S,K):
        sum_numarator = sum(sum(S[0:K,0:K]))
        sum_denominator = sum(sum(S))
        variance_retained = sum_numarator/sum_denominator
        return variance_retained
