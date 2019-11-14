import data.pca_model.PCA as m_pca
import data.pca_model.config as cfg
import numpy as np

class Model:
    def __init__(self, X):
        self.X = X

    def reduce_dimension(self):
        X_norm, mu, sigma = m_pca.PCA.featureNormalize(self.X)
        #modifying X using random projection since calculation of covariance matrix is not possible
        [m,n] = X_norm.shape
        rand_arr = np.random.rand(n,cfg.K)
        Z = X_norm @ rand_arr
        print(Z.shape)
        # U, self.S = m_pca.PCA.pca(X_norm)
        
        # Z = m_pca.PCA.projectData(X_norm, U, cfg.K)
        return Z

    # def getVarianceRetained(self):
    #     variance_retained = m_pca.PCA.varianceRetained(self.S, cfg.K)
    #     return variance_retained