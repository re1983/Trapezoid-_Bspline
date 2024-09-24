import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

# 定义控制点
p1 = (-5, 20)
p2 = (-15, -10)
p3 = (5, 20)
p4 = (10, -15)

# 计算中点和质心
p1_2 = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
p3_4 = ((p3[0] + p4[0]) / 2, (p3[1] + p4[1]) / 2)
p1_3 = ((p1[0] + p3[0]) / 2, (p1[1] + p3[1]) / 2)
p2_4 = ((p2[0] + p4[0]) / 2, (p2[1] + p4[1]) / 2)

# 计算质心（质心坐标）
centroid_x = (p1[0] + p2[0] + p3[0] + p4[0]) / 4
centroid_y = (p1[1] + p2[1] + p3[1] + p4[1]) / 4

# 提取 x 和 y 坐标
x1, y1 = p1
x2, y2 = p2
x3, y3 = p3
x4, y4 = p4

# 应用鞋带公式计算面积
area = 0.5 * abs(x1 * y2 + x2 * y3 + x3 * y4 + x4 * y1 - (y1 * x2 + y2 * x3 + y3 * x4 + y4 * x1))

# 定义控制点
ctrlpts = [
    [p2[0], p2[1], 0.0], [p1_2[0], p1_2[1], 0.0], [p1[0], p1[1], 0.0],
    [p2_4[0], p2_4[1], 0.0], [centroid_x, centroid_y, area], [p1_3[0], p1_3[1], 0.0],
    [p4[0], p4[1], 0.0], [p3_4[0], p3_4[1], 0.0], [p3[0], p3[1], 0.0]
]

# 转换控制点为 NumPy 数组并分离坐标
ctrlpts = np.array(ctrlpts)
x_ctrl = ctrlpts[:, 0]
y_ctrl = ctrlpts[:, 1]
z_ctrl = ctrlpts[:, 2]

# 确保 x 和 y 坐标是严格递增的
unique_x = np.unique(x_ctrl)
unique_y = np.unique(y_ctrl)

# 生成 Z 值的二维数组 (3x3)
Z_array = np.zeros((len(unique_y), len(unique_x)))

# 填充 Z 值
for i, x in enumerate(unique_x):
    for j, y in enumerate(unique_y):
        # 找到对应的 Z 值
        idx = np.where((x_ctrl == x) & (y_ctrl == y))
        if idx[0].size > 0:
            Z_array[j, i] = z_ctrl[idx[0][0]]
        else:
            Z_array[j, i] = np.nan  # 如果没有对应的 Z 值，设置为 NaN

# 使用 RectBivariateSpline 生成样条曲面
spline = RectBivariateSpline(unique_y, unique_x, Z_array)

# 生成细化网格
x_fine = np.linspace(unique_x.min(), unique_x.max(), 100)
y_fine = np.linspace(unique_y.min(), unique_y.max(), 100)
X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
Z_fine = spline(y_fine, x_fine)  # 注意 y_fine 和 x_fine 的顺序

# 绘制 3D 曲面
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制表面
ax.plot_surface(X_fine, Y_fine, Z_fine, cmap='viridis', edgecolor='none')

# 绘制控制点
ax.scatter(x_ctrl, y_ctrl, z_ctrl, color='red', label='Control Points')  # 控制点
ax.set_title('3D B-Spline Surface')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.legend()

plt.show()
