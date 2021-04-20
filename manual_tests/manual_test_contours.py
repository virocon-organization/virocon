import numpy as np

import matplotlib.pyplot as plt

from virocon.contours import sort_points_to_form_continuous_line

# %% sort_points_to_form_continuous_line

phi = np.linspace(0, 2 * np.pi, num=10, endpoint=False)
ref_x = np.cos(phi)
ref_y = np.sin(phi)
plt.close("all")
plt.figure()
plt.plot(ref_x, ref_y)

rng = np.random.default_rng()
rand_idx = np.arange(len(ref_x))
rng.shuffle(rand_idx)

rand_x = ref_x[rand_idx]
rand_y = ref_y[rand_idx]
plt.plot(rand_x, rand_y)

my_x, my_y = sort_points_to_form_continuous_line(rand_x, rand_y, search_for_optimal_start=True)

plt.plot(my_x, my_y)

np.testing.assert_array_equal(my_x, ref_x)
np.testing.assert_array_equal(my_y, ref_y)




