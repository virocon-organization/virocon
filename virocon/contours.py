import numpy as np
import scipy.stats as sts


from virocon._n_sphere import NSphere

def calculate_alpha(state_duration, return_period):
    alpha = state_duration / (return_period * 365.25 * 24)
    return alpha


class IFORMContour():
    
    def __init__(self, model, alpha, n_points=180):
        self.model = model
        self.alpha = alpha
        self.n_points = n_points
        
        self._compute()
        
        
    def _compute(self,):
        """
        Calculates coordinates using IFORM.

        """
        n_dim = self.model.dimensions
        n_points = self.n_points
        
        beta = sts.norm.ppf(1 - self.alpha)
        self.beta = beta

        # TODO Update NSphere to handle n_dim case with order
        # Create sphere
        if n_dim == 2:
            _phi = np.linspace(0, 2 * np.pi , num=n_points, endpoint=False)
            _x = np.cos(_phi)
            _y = np.sin(_phi)
            _circle = np.stack((_x,_y), axis=1)
            sphere_points = beta * _circle

        else:
            sphere = NSphere(dim=n_dim, n_samples=n_points)
            sphere_points = beta * sphere.unit_sphere_points

        # Get probabilities for coordinates
        norm_cdf = sts.norm.cdf(sphere_points)

        # Inverse procedure. Get coordinates from probabilities.
        coordinates = self.model.icdf(norm_cdf)
        # for index, distribution in enumerate(distributions):
        #     data[index] = distribution.i_cdf(norm_cdf_per_dimension[index], rv_values=data,
        #                                      dependencies=self.distribution.dependencies[index])


        self.sphere_points = sphere_points
        self.coordinates = coordinates
