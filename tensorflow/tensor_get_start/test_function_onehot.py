# -*- coding: utf-8 -*-
# @Author: Teiei
# @Date:   2018-07-26 22:37:01
# @Last Modified by:   Teiei
# @Last Modified time: 2018-07-27 01:20:26
import numpy as np
import pytest
def dense_to_one_hot(labels_dense, num_classes=10):
	"""Convert class labels from scalars to one-hot vectors."""
	num_labels = labels_dense.shape[0]
	index_offset = np.arange(num_labels) * num_classes
	labels_one_hot = np.zeros((num_labels, num_classes))
	labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1  ## 调用Numpy的flat方法、ravel方法
	return labels_one_hot
## method 2 to define one_hot convert 
'''
def ref_one_hot(x, shape):
    result = np.zeros((x.shape[0],) + shape)
    result[np.arange(x.shape[0]), x.flatten()] = 1
    return result
'''
'''
	x 是一个一维向量的话，如[0,3,7]那么shape就是8 ,即max(x)+1
	x 是一个矩阵呢？
'''
def ref_one_hot(x, shape):
	print("----enter ref_one_hot --------")
	print("x = ",x)
	print("shape = ",shape)
	x0_shape = (x.shape[0],) + shape  ##(10,10)
	print("x0_shape = {},type(x0_shape)={}".format(x0_shape,type(x0_shape)))
	result = np.zeros((x.shape[0],) + shape)  ## 做一个x.shape[0] * shape的矩阵
	print("## result =",result)
	print("---------------")
	arrange_shape = np.arange(x.shape[0]) ## 其实这里就是传进来的x ,只不过要确保它是np对象
	print("arrange_shape = {},type(arrange_shape)={}".format(arrange_shape,type(arrange_shape)))
	x_flatten = x.flatten() 
	print("x_flatten = {},type(x_flatten)={}".format(x_flatten,type(x_flatten)))
	result[np.arange(x.shape[0]), x.flatten()] = 1
	print("-----return  ref_one_hot ----------")
	return result
'''
### test use @pytest.mark.parametrize
@pytest.mark.parametrize("seed", [313])
@pytest.mark.parametrize("inshape", [(100, 1)])
@pytest.mark.parametrize("shape", [(10,)])  ## shape is a tuple
def test_one_hot_forward(seed, inshape, shape,): ## inshape is also a tulpe
    rng = np.random.RandomState(seed)
    print("### shape = {},shape[0] ={},type(shape)={}".format(shape,shape[0],type(shape)))
    print("### inshape = {},type(inshape)={}".format(inshape,type(inshape)))
    input = rng.randint(0, shape[0], size=inshape)  ### preduce size个数据(可以是矩阵),数据取值为(1~shape[0])
    print('intput = {},type ={}'.format(input,type(input)))					## 即0 ~ 10
    print("### input.shape =\n",input.shape)
    r = ref_one_hot(input, shape)
    print("---------- pytest ,r = \n",r)

def test_one_hot_forward_no_parametrize():
	print("-------test_one_hot_forward_no_parametrize----------")
	rng = np.random.RandomState(313)
	shape = (10,)
	print('shape = {},type(shape)={}'.format(shape,type(shape)))
	print("shape[0] = ",shape)
	inshape = (100, 1)
	print("inshape = ",inshape)
	input = rng.randint(0, shape[0], size=inshape)
	print('intput = {},type ={}'.format(input,type(input)))
	r = ref_one_hot(input, shape)
	print("### r = ",r)
'''
if __name__ == '__main__':
	test_ndarray = np.arange(10)
	print("test_ndarray = {},type(test_ndarray) = {}".format(test_ndarray,type(test_ndarray)))
	print("test_ndarray.shape = {}".format(test_ndarray.shape))

	print("-------method1----------")
	onehot_ndarray = dense_to_one_hot(test_ndarray,len(test_ndarray))
	print(onehot_ndarray)

	print("-------method2-1----------")
	print("test_ndarray = {},type(test_ndarray) = {}".format(test_ndarray,type(test_ndarray)))
	print("test_ndarray.shape = {}".format(test_ndarray.shape))
	onehot_ndarray2 = ref_one_hot(test_ndarray,test_ndarray.shape)
	print("onehot_ndarray2 = ",onehot_ndarray2)

	print("-------method2-2----------")
	x3 = np.array([0,3,7])
	print("x3 = ",x3)
	print("x3 = {},type(x3) = {}".format(x3,type(x3)))
	onehot_ndarray3 = ref_one_hot(x3,(8,))
	print("onehot_ndarray3 = ",onehot_ndarray3)

	print("-------method2-3----------")
	x3 = np.array([[0,3,4],[7,8,9]])
	print("x3 = {},type(x3) = {}".format(x3,type(x3)))
	onehot_ndarray3 = ref_one_hot(x3,(8,))
	print("onehot_ndarray3 = ",onehot_ndarray3)



