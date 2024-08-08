import numpy as np
import matplotlib.pyplot as plt

# 参数设置
theta = np.radians(30)  # 30度旋转
w = 4
h = 2

# 缩放协方差矩阵
Sigma_scale = np.array([[w**2 / 4, 0],
                        [0, h**2 / 4]])

# 旋转矩阵
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

# 构建协方差矩阵
Sigma_g = R @ Sigma_scale @ R.T

# 计算逆矩阵
Sigma_g_inv = np.linalg.inv(Sigma_g)

# 打印逆矩阵
print("Covariance Matrix (Sigma_g):")
print(Sigma_g)

print("\nInverse Covariance Matrix (Sigma_g_inv):")
print(Sigma_g_inv)

# 创建二维网格
x, y = np.mgrid[-3:3:.01, -3:3:.01]
pos = np.dstack((x, y))

# 二维高斯分布的概率密度
def mahalanobis_distance(x, inv_cov):
    return np.sqrt(np.sum(x @ inv_cov * x, axis=-1))

# 计算马氏距离
mdist = mahalanobis_distance(pos, Sigma_g_inv)

# 绘制马氏距离等高线
plt.figure(figsize=(8, 6))
plt.contour(x, y, mdist, levels=np.arange(0, 5, 0.5), cmap='viridis')
plt.title('Mahalanobis Distance Contour with Inverse Covariance Matrix')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.colorbar(label='Mahalanobis Distance')
plt.axhline(0, color='gray', lw=0.5)
plt.axvline(0, color='gray', lw=0.5)
plt.grid(True, linestyle='--', alpha=0.7)
plt.axis('equal')
plt.show()