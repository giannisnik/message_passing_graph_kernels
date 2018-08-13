import numpy as np
import scipy.sparse as sp
from scipy.linalg import svd
from sklearn.utils import check_random_state

class Nystrom():
    
	def __init__(self, n_components=100, random_state=None):
		self.n_components = n_components
		self.random_state = random_state

	def fit(self, hists, y=None):
		rnd = check_random_state(self.random_state)
		n_samples = len(hists)

		# get basis vectors
		if self.n_components > n_samples:
			n_components = n_samples
		else:
			n_components = self.n_components
		n_components = min(n_samples, n_components)
		inds = rnd.permutation(n_samples)
		basis_inds = inds[:n_components]
		basis = []
		for ind in basis_inds:
			basis.append(hists[ind])

		basis_kernel = np.zeros((n_components, n_components))
		for i in range(n_components):
			for j in range(i,n_components):
				for n in basis[i]:
					if n in basis[j]:
						basis_kernel[i,j] += min(basis[i][n],basis[j][n])
				basis_kernel[j,i] = basis_kernel[i,j]

		# sqrt of kernel matrix on basis vectors
		U, S, V = svd(basis_kernel)
		S = np.maximum(S, 1e-12)
		self.normalization_ = np.dot(U * 1. / np.sqrt(S), V)
		self.components_ = basis
		self.component_indices_ = inds
		return self

	def transform(self, hists):
		embedded = np.zeros((len(hists), len(self.components_)))
		for i in range(len(hists)):
			for j in range(len(self.components_)):
				for n in hists[i]:
					if n in self.components_[j]:
						embedded[i,j] += min(hists[i][n], self.components_[j][n])
				
		return np.dot(embedded, self.normalization_.T)

