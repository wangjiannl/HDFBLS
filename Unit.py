import numpy as np

class ILFR:
    def __init__(self, in_dim, num_sam, center, num_fuzzy_set=5, 
                 center_min=0.0, center_max=1.0):
        self.in_dim = int(in_dim)
        self.num_sam = int(num_sam)
        self.num_fuzzy_set = int(num_fuzzy_set)
        self.centers = center

    def fuzzify_input(self, X, spread=None):
        """
        X: (N, M)  ->  UL: (N, M, K)
        UL[n,i,k] = exp( -(X[n,i]-c_k)^2 / (2*sigma^2) )
        """
        X = np.asarray(X, dtype=np.float64)
        if X.shape != (self.num_sam, self.in_dim):
            raise ValueError(f"X shape {X.shape} != (N,M)=({self.num_sam},{self.in_dim})")

        N, M, K = self.num_sam, self.in_dim, self.num_fuzzy_set

        UL = np.empty((N, M, K), dtype=np.float64)
        denom = 2.0 * (spread ** 2)
        for k in range(K):
            ck = self.centers[k]
            diff = X - ck
            UL[:, :, k] = np.exp(-(diff * diff) / denom)
        return UL

    def compute_activation(self, UL, CL, DL):
        """
        :param UL: shape (N, M, K)
        return: activation matrix V of shape (N, M)
        """
        V = np.ones((self.num_sam, self.in_dim), dtype=np.float64)
        for i in range(self.in_dim):
            if np.all(CL[i, :] == 0):
                DL[i] = 1

            if DL[i] == 1:
                V[:, i] = 1.0

            else:
                prod = np.ones(self.num_sam, dtype=np.float64)
                for k in range(self.num_fuzzy_set):
                    if CL[i, k] == 1:
                        prod *= (1 - UL[:, i, k])

                V[:, i] = 1 - prod

        H = np.prod(V, axis=1) 
        return H