#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample (almost) equally distributed points on an n-spheres surface.
"""

import itertools

import numpy as np

__all__ = ["NSphere"]


class NSphere():
    """
    This class calculates almost equally spaced points on a n-sphere.

    It considers the Thomson problem [#]_ to distribute the points on the sphere.

    References
    ----------
    .. [#] https://en.wikipedia.org/wiki/Thomson_problem

    Examples
    --------

    >>> sphere = NSphere(3, 1000)
    >>> samples = sphere.unit_sphere_points
    >>> #%% plot sphere
    >>> from mpl_toolkits.mplot3d import Axes3D
    >>> import matplotlib.pyplot as plt
    >>> plt.close('all')
    >>> #fig = plt.figure(figsize=plt.figaspect(1))
    >>> #ax = fig.add_subplot(111, projection='3d')
    >>> u = np.linspace(0, 2 * np.pi, 50)
    >>> v = np.linspace(0, np.pi, 30)
    >>> x = np.outer(np.cos(u), np.sin(v))
    >>> y = np.outer(np.sin(u), np.sin(v))
    >>> z = np.outer(np.ones(np.size(u)), np.cos(v))
    >>> #wireframe_plot = ax.plot_wireframe(x, y, z, rcount=10, ccount=10,)
    >>> #sphere_plot = ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2],\
                                 c='green', depthshade=False, label="verteilt")
    >>> #plot_legend = ax.legend()

    """

    def __init__(self, dim, n_samples):
        """
        Parameters
        ----------
        dim : int
            The number of dimensions. (i.e. the n in n-sphere plus 1)
        n_samples : int
            The number of points to distribute on the n-sphere.

        """
        self.dim = dim
        self.n_samples = n_samples

        #combinations for selecting all different indice
        self.combs = np.array([comb for comb in itertools.combinations(range(self.n_samples), 2)])

        self.unit_sphere_points = self._random_unit_sphere_points()
        self.init_e_pot = self._pot_energy()
        self._relax_points()
        self.rela_e_pot = self._pot_energy()
        self.improvement = (self.init_e_pot - self.rela_e_pot) / self.init_e_pot * 100


    def _relax_points(self,):
        """
        Iteratively move points to reduce potential energy.

        Iteratively moves points in the direction of the tangential part of the
        coloumb forces and calculates the potential energy after each iteration.
        The best state, i.e. the state with least potential energy, is saved.
        """

        best_state = np.copy(self.unit_sphere_points)
        best_pot = self._pot_energy()

        # Define the number osf iterations based on the size of the sample size.
        # At least 10 iterations should be performed. 10 is an empirical value.
        max_iters = max([10, int(10000 / self.n_samples)])


        for iteration in range(1, max_iters):

            # Tau controls the step size of the optimization. With each iteration
            # a smaller step is chosen. 3 is an empirical value.
            tau = 3 / iteration

            tang_forces = self._tangential_forces(self._get_forces())

            # Norm to max force
            max_force = np.max(np.linalg.norm(tang_forces, axis=1,))
            tang_forces /= max_force

            self.unit_sphere_points += tang_forces * tau
            self.unit_sphere_points /= np.linalg.norm(self.unit_sphere_points,
                                                      axis=1,
                                                      keepdims=True)

            curr_pot = self._pot_energy()
            if curr_pot < best_pot:
                best_state = np.copy(self.unit_sphere_points)
                best_pot = curr_pot

        self.unit_sphere_points = best_state

    def _random_unit_sphere_points(self):
        """
        Generates equally distributed points on the sphere's surface.

        Note
        ----
        Used algorithm:

        - Use a N(0,1) standard normal distribution to generate cartesian coordinate vectors.
        - Normalize the vectors to length 1.
        - The points then are uniformly distributed on the sphere's surface.

        """
        # create pseudorandom number generator with seed for reproducability
        prng = np.random.RandomState(seed=43)
        #  draw normally distributed samples
        rand_points = prng.normal(size=(self.n_samples, self.dim))
        # calculate lengths of vectors
        radii = np.linalg.norm(rand_points, axis=1, keepdims=True)
        # normalize
        return rand_points / radii


    def _pot_energy(self, ):
        """
        Calculates the potential energy of the current state.

        Assume the points on the sphere are electrons with charge equal to 1 and
        assume Coulomb's constant equal to 1. Then the electrostatic potential energy of
        :math:`N` electrons can be written as:

        .. math::

            U = \\sum_{i < j } \\frac{1}{\\lvert {r_{ij}} \\rvert}

        With :math:`0 \\leq i < j \\leq N` and :math:`\\lvert {r_{ij}} \\rvert` the distance vector
        between electron :math:`i` and :math:`j`.

        """

        dist_vectors = (self.unit_sphere_points[self.combs[:, 0]]
                        - self.unit_sphere_points[self.combs[:, 1]])

        distances = np.linalg.norm(dist_vectors, axis=1, keepdims=True)

        return np.sum(distances**-1)

    def _get_forces(self,):
        """
        Calculates the Coloumb forces.

        Assume the :math:`N` points are electrons with charge 1 and Coloumb's constant is 1.
        The Coulomb force on point :math:`i` can then be expressed as:

        .. math::

            F_i = \\sum_{j \\neq i} \\frac{1}{{\\lvert {r_{ij}} \\rvert}^2},
            0 \\leq j \\leq N
        """
        r_dash = self.unit_sphere_points[:, np.newaxis, :] - self.unit_sphere_points
        with np.errstate(divide='ignore', invalid='ignore'):
            single_forces = r_dash / (np.linalg.norm(r_dash, axis=2, keepdims=True)**3)
        return np.nansum(single_forces, axis=1)


    def _tangential_forces(self, forces):
        """
        Calculates the tangential part of the forces.

        First calculates the radial forces and substracts them to get the tangential part.
        With :math:`\\vec{r_i}` the radial vector pointing to point :math:`i`,
        :math:`F_i` the force on point :math:`i`, and scalar product :math:`\\langle~,~\\rangle`,
        the radial force can be calculated like:

            .. math::

                \\vec{F_{i_{rad}}} = \\langle\\vec{F_i}, \\vec{r_i}\\rangle  \\vec{r_i}

        The tangential force is then expressed by:

            .. math::

                \\vec{F_{i_{tan}}} = \\vec{F_i} - \\vec{F_{i_{rad}}}


        Parameters
        ----------
        forces : ndarray,
            The forces to calculate the tangential part of.

        """

        scalar_product = np.sum(forces * self.unit_sphere_points, axis=1, keepdims=True)
        radial_forces = scalar_product * self.unit_sphere_points
        return forces - radial_forces


if __name__ == "__main__":

#    sphere = NSphere(3, 1000)
#
#    samples = sphere.unit_sphere_points
#
#    print("Initial : E_pot = {}".format(sphere.init_e_pot))
#    print("Relaxed : E_pot = {}".format(sphere.rela_e_pot))
#    print("Verbesserung: {} %".format(sphere.improvement))
#
#    #%% plot sphere
#    from mpl_toolkits.mplot3d import Axes3D
#    import matplotlib.pyplot as plt
#    plt.close('all')
#    fig = plt.figure(figsize=plt.figaspect(1))
#    ax2 = fig.add_subplot(111, projection='3d')
#
#    u = np.linspace(0, 2 * np.pi, 50)
#    v = np.linspace(0, np.pi, 30)
#
#    x = np.outer(np.cos(u), np.sin(v))
#    y = np.outer(np.sin(u), np.sin(v))
#    z = np.outer(np.ones(np.size(u)), np.cos(v))
#
#    ax2.plot_wireframe(x, y, z, rcount=10, ccount=10,)
#
#
#    ax2.scatter(samples[:, 0], samples[:, 1], samples[:, 2],
#                c='green', depthshade=False, label="verteilt")
#
#    ax2.legend()
#    plt.show()
    import doctest
    doctest.testmod()