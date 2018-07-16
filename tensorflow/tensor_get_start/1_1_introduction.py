## 这段很短的 Python 程序生成了一些三维数据, 然后用一个平面拟合它

import tensorflow as tf
import numpy as np

# 使用 NumPy 生成假数据(phony data), 总共 100 个点.
## np.random.rand(2, 100) 产生一个2*100的矩阵，矩阵的内容貌似都是0~1直接的值。
## np.float32 :
x_data = np.float32(np.random.rand(2, 100)) # 随机输入
y_data = np.dot([0.100, 0.200], x_data) + 0.300  ## dot矩阵相乘

# 构造一个线性模型
# 
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, x_data) + b  ## matmul 就是 y = WX+b

# 最小化方差
loss = tf.reduce_mean(tf.square(y - y_data))  ## reduce_mean 是求平均数
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化变量
init = tf.initialize_all_variables()

# 启动图 (graph)
sess = tf.Session()
sess.run(init)

# 拟合平面
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print( step, sess.run(W), sess.run(b) )  ## train的过程就是不断调整W和b使得loss最小的过程

# 得到最佳拟合结果 W: [[0.100  0.200]], b: [0.300]