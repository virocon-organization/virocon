import numpy as np

import matplotlib.pyplot as plt

from virocon.utils import sort_points_to_form_continuous_line

# %% sort_points_to_form_continuous_line

phi = np.linspace(0, 2 * np.pi, num=10, endpoint=False)
ref_x = np.cos(phi)
ref_y = np.sin(phi)
plt.close("all")
plt.figure()
plt.plot(ref_x, ref_y)

shuffle_idx = [5, 2, 0, 6, 9, 4, 1, 8, 3, 7]

rand_x = ref_x[shuffle_idx]
rand_y = ref_y[shuffle_idx]
plt.plot(rand_x, rand_y)

my_x, my_y = sort_points_to_form_continuous_line(rand_x, rand_y, search_for_optimal_start=True)

plt.plot(my_x, my_y)
plt.show()

np.testing.assert_array_equal(my_x, ref_x)
np.testing.assert_array_equal(my_y, ref_y)




