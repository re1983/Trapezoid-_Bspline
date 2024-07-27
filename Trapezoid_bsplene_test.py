from geomdl import BSpline
from geomdl.visualization import VisMPL
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
# Control points
# ctrlpts = [
#     [[-25.0, -25.0, -10.0], [-25.0, -15.0, -5.0], [-25.0, -5.0, 0.0], [-25.0, 5.0, 0.0], [-25.0, 15.0, -5.0], [-25.0, 25.0, -10.0]],
#     [[-15.0, -25.0, -8.0], [-15.0, -15.0, -4.0], [-15.0, -5.0, -4.0], [-15.0, 5.0, -4.0], [-15.0, 15.0, -4.0], [-15.0, 25.0, -8.0]],
#     # [[-5.0, -25.0, -5.0], [-5.0, -15.0, -3.0], [-5.0, -5.0, -8.0], [-5.0, 5.0, -8.0], [-5.0, 15.0, -3.0], [-5.0, 25.0, -5.0]],
#     # [[5.0, -25.0, -3.0], [5.0, -15.0, -2.0], [5.0, -5.0, -8.0], [5.0, 5.0, -8.0], [5.0, 15.0, -2.0], [5.0, 25.0, -3.0]],
#     # [[15.0, -25.0, -8.0], [15.0, -15.0, -4.0], [15.0, -5.0, -4.0], [15.0, 5.0, -4.0], [15.0, 15.0, -4.0], [15.0, 25.0, -8.0]],
#     [[25.0, -25.0, -10.0], [25.0, -15.0, -5.0], [25.0, -5.0, 2.0], [25.0, 5.0, 2.0], [25.0, 15.0, -5.0], [25.0, 25.0, -10.0]]
# ]

p1 = -5, 20
p2 = -15, -10
p3 = 5, 20
p4 = 10 , -15

p1_2 = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
p3_4 = ((p3[0] + p4[0]) / 2, (p3[1] + p4[1]) / 2)
p1_3 = ((p1[0] + p3[0])/2 , (p1[1] + p3[1])/2)
p2_4 = ((p2[0] + p4[0]) / 2, (p2[1] + p4[1]) / 2)
# Calculate the centroid (center of mass)
centroid_x = (p1[0] + p2[0] + p3[0] + p4[0]) / 4
centroid_y = (p1[1] + p2[1] + p3[1] + p4[1]) / 4

# Extract x and y coordinates
x1, y1 = p1
x2, y2 = p2
x3, y3 = p3
x4, y4 = p4

# Apply the Shoelace formula
area = 0.5 * abs(x1*y2 + x2*y3 + x3*y4 + x4*y1 - (y1*x2 + y2*x3 + y3*x4 + y4*x1))

# print("Area of the quadrilateral:", area)

# centroid = (centroid_x, centroid_y)
# print("centroid: ", centroid)


ctrlpts = [
    [[p2[0], p2[1], 0.0], [p1_2[0], p1_2[1], 0.0], [p1[0], p1[1], 0.0]],
    # [[p2_4[0], p2_4[1], 0.0], [centroid_x, centroid_y, (1/area)], [p1_3[0], p1_3[1], 0.0]],
    [[p2_4[0], p2_4[1], 0.0], [centroid_x, centroid_y, area], [p1_3[0], p1_3[1], 0.0]],
    [[p4[0], p4[1], 0.0], [p3_4[0], p3_4[1], 0.0], [p3[0], p3[1], 0.0]],
]


# ctrlpts = [
#     [[-10.0, -10.0, 0.0], [-10.0, 10.0, 0.0]],
#     [[0.0, 0.0, 10.0]],
#    [[10.0, -10.0, 0.0], [10.0, 10.0, 0.0]],
# ]

# Create a BSpline surface
surf = BSpline.Surface()

# Set degrees
surf.degree_u = 2
surf.degree_v = 2

# Set control points
surf.ctrlpts2d = ctrlpts

# # Set knot vectors
# surf.knotvector_u = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0]
# surf.knotvector_v = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0]

# Set knot vectors
from geomdl import utilities
surf.knotvector_u = utilities.generate_knot_vector(surf.degree_u, surf.ctrlpts_size_u)
# print("knotvector_u: ", surf.knotvector_u)
surf.knotvector_v = utilities.generate_knot_vector(surf.degree_v, surf.ctrlpts_size_v)
# print("knotvector_v: ", surf.knotvector_v)


# Set evaluation delta
surf.delta = 0.025

# Evaluate surface points
surf.evaluate()

# Import and use Matplotlib's colormaps
from matplotlib import cm

# Plot the control points grid and the evaluated surface
surf.vis = VisMPL.VisSurface()
surf.render(colormap=cm.cool)


# Extract evaluated points
points = np.array(surf.evalpts)
x = points[:, 0].reshape((surf.sample_size_u, surf.sample_size_v))
y = points[:, 1].reshape((surf.sample_size_u, surf.sample_size_v))
z = points[:, 2].reshape((surf.sample_size_u, surf.sample_size_v))

# Plot using Matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='cool')

# Define the XY plane
x_plane, y_plane = np.meshgrid(np.linspace(np.min(x), np.max(x), 10), np.linspace(np.min(y), np.max(y), 10))
z_plane = np.zeros_like(x_plane)  # Set Z to 0 for the XY plane

# Plot the XY plane
ax.plot_surface(x_plane, y_plane, z_plane, color='gray', alpha=0.5)

# Display the plot
plt.show()


