## ref :  https://blog.csdn.net/u012609509/article/details/70230204
import numpy as np
print('------1. multiply(或者*) 为对应元素相乘  -------')
# 2-D array: 2 x 3
two_dim_matrix_one = np.array([[1, 2, 3], [4, 5, 6]])
another_two_dim_matrix_one = np.array([[7, 8, 9], [4, 7, 1]])

# 对应元素相乘 element-wise product
element_wise = two_dim_matrix_one * another_two_dim_matrix_one
print('element wise product(multiply):\n %s' %(element_wise))

# 对应元素相乘 element-wise product
element_wise_2 = np.multiply(two_dim_matrix_one, another_two_dim_matrix_one)
print('element wise product(*):\n %s' % (element_wise_2))

print('------ 2.  dot 为矩阵相乘 -------')
import numpy as np

# 2-D array: 2 x 3
two_dim_matrix_one = np.array([[1, 2, 3], [4, 5, 6]])
# 2-D array: 3 x 2
two_dim_matrix_two = np.array([[1, 2], [3, 4], [5, 6]])

two_multi_res = np.dot(two_dim_matrix_one, two_dim_matrix_two)
print('two_multi_res(dot):\n %s' %(two_multi_res))

# 1-D array
'''
one_dim_vec_one = np.array([1, 2, 3])
one_dim_vec_two = np.array([4, 5, 6])
one_result_res = np.dot(one_dim_vec_one, one_dim_vec_two)
print('one_result_res:\n %s' %(one_result_res))
'''

