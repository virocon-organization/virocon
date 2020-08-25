"""
Utilities for contours.
"""
import numpy as np
from sklearn.decomposition import PCA as skPCA


class PCA():
    """
    Principal component analysis (PCA) for 2D wave data.
    
    Wraps sklearn PCA to ease use for wave data.
    Similar to ESSC PCA.
    https://github.com/WEC-Sim/WDRT/blob/master/WDRT/ESSC.py
    
    """
    
    def __init__(self, data):
        """

        Parameters
        ----------
        data : ndarray
            The wave data with shape = (number of observations, 2).

        """
        pca = skPCA(n_components=2)
        pca.fit(data)
        self.components = pca.components_
        self.transformed = self._transform(data)
        
    def _transform(self, data):
        #  sklearn pca.transform normalizes before transforming 
        # X_transformed = np.dot(X, self.components_.T)
        # here we do not normalize
        pca_components = np.dot(data, self.components.T)

        # This is a point where we divert from ESSC.
        # ESSC always uses a shift, while we only use it if necessary.
        # shift = abs(min(pca_components[:, 1])) + 0.1 # ESSC style shift
        c2_min = min(pca_components[:, 1])
        if c2_min < 0:
            self.shift = abs(c2_min) + 0.1  # Calculate shift
        else:
            self.shift = 0
            
        # Apply shift to Component 2 to make all values positive
        pca_components[:, 1] = pca_components[:, 1] + self.shift
        return pca_components
    
    def inverse_transform(self, data, clip=True):
        """
        Returns the inverse pca of data.        

        Parameters
        ----------
        data : ndarray
            The data to inverse transform.
        clip : boolean, optional
            If True unphysical values(<0) are set to zero. The default is True.

        Returns
        -------
        ndarray
            The inverse transformed data.

        """
        dat = data.copy()
        dat[:,1] = dat[:,1] - self.shift
        inv_transformed_data = np.dot(dat, self.components)
        if clip:
            return np.clip(inv_transformed_data, 0, None)
        else:
            return inv_transformed_data
