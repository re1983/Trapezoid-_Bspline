import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

# 定义控制点网格
x = np.linspace(0, 5, 6)  # 控制点在 x 方向的取值
y = np.linspace(0, 5, 6)  # 控制点在 y 方向的取值

# 创建一个二维控制点矩阵（可以根据需要调整）
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)  # 控制点的 z 值，可以更改为其他函数

# 使用 RectBivariateSpline 生成样条曲面
spline = RectBivariateSpline(x, y, Z)

# 生成细化网格
x_fine = np.linspace(0, 5, 100)
y_fine = np.linspace(0, 5, 100)
X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
Z_fine = spline(x_fine, y_fine)

# 绘制 3D 曲面
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制表面
ax.plot_surface(X_fine, Y_fine, Z_fine, cmap='viridis', edgecolor='none')

# 绘制控制点
ax.scatter(X, Y, Z, color='red', label='Control Points')  # 控制点
ax.set_title('3D B-Spline Surface')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.legend()

plt.show()
