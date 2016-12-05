# coding=utf-8
from numpy import *
import matplotlib.pyplot as plt


# --------------------------------
# 这是一个函数集，被test_kmeans.py调用
# 单词解释
# centroids>>聚类中心
# distance>>距离，当前特指欧氏距离
# cluster>>类，聚类簇，集群，集合
# dim>>dimension>>次元，度，维
# --------------------------------

# 计算欧氏距离
def euclDistance(vector1, vector2):
	# power函数计算乘方
	# sqrt函数平方根
	return sqrt(sum(power(vector2 - vector1, 2)))


# 初始化聚类中心，dataSet为样本数据矩阵，参数k为聚类中心数目
def initCentroids(dataSet, k):
	# 根据传入的矩阵获取矩阵的行列数,行数就是numSample样本总数
	# 列数赋值给dim>>dimension>>维数
	numSamples, dim = dataSet.shape
	# 创建一个0数组(矩阵)大小为k*dim
	centroids = zeros((k, dim))
	for i in range(k):
		# 随机产生从0到numSample的实数,类型强制转化为整型
		index = int(random.uniform(0, numSamples))
		# 把index作为索引来取出dataSet的第index行
		# 即随机抽取index行作为初始的聚类中心，放到k*dim的矩阵中
		centroids[i, :] = dataSet[index, :]
	# 得到的centroids就是随机抽取出的聚类中心矩阵
	# 返回一个初始的随机聚类中心
	return centroids


# 开始k-means聚类，dataSet为样本数据矩阵，参数k为聚类中心数目
def k_means(dataSet, k):
	# 样本的数量
	numSamples = dataSet.shape[0]
	# first column stores which cluster this sample belongs to,本示例所属的第一列存储
	# second column stores the error between this sample and its centroid第二列存储此示例和它的聚类中心之间的错误
	# 做一个样本数*2的二维零矩阵。cluster(簇，聚集)
	# clusterAssment就是一个(数据索引>>欧氏距离)的存储器
	clusterAssment = mat(zeros((numSamples, 2)))
	# 设置变量clusterChanged为真
	clusterChanged = True
	# 步骤1：初始化聚类中心，执行初始化聚类中心函数，得到随机抽取的聚类中心
	centroids = initCentroids(dataSet, k)
	# 判断clusterChanged如果为真
	while clusterChanged:
		# 首先把clusterChanged设置为假
		clusterChanged = False
		# 迭代每一个样本，迭代次数为numSample(样本数目)
		for i in xrange(numSamples):
			# distance>>距离，minDist>>最小距离
			# 定义最小欧氏距离为100000.0
			minDist = 100000.0
			# 初始化最小索引
			minIndex = 0
			# 步骤2：对一条数据，迭代每一个聚类中心，找到该数据最接近的聚类中心
			for j in range(k):
				# 执行计算欧氏距离函数
				# 传入第一个参数vector1为随机聚类中心第j行的数组，vector2为数据矩阵抽取的一行
				distance = euclDistance(centroids[j, :], dataSet[i, :])
				# 判断如果计算得到的欧氏距离小于定义的最小欧氏距离
				if distance < minDist:
					# 更新定义的最小欧氏距离
					minDist = distance
					# 更新最小索引为当前符合条件的聚类中心的索引
					# minIndex就是当前数据条目所属的聚类
					minIndex = j
			# 步骤3：更新数据所属的聚类信息
			if clusterAssment[i, 0] != minIndex:
				# 符合上面条件的话，重新定义clusterChanged为真，触发下次while循环
				clusterChanged = True
				# 把clusterChanged当前行第一个值定为minIndex，即循环后最终确定下来的所属聚类集
				# 把clusterChanged当前行第二个值定为minDist的平方，即最终最小欧氏距离的平方
				clusterAssment[i, :] = minIndex, minDist ** 2
		# 上面已经把所有数据循环一遍
		
		# 步骤4：更新聚类中心
		for j in range(k):
			'''
			# 关键代码解析，从内到外~~~
			# 使用取出的第0个下标取出dataSet中对应位置的原始数据
			dataSet[
				# 获取到布尔数组里不为0的元素的下标，返回值的0下标对应的元素为所需的下标集
				# nonzero函数>>返回数组a中值不为零的元素的下标，它的返回值是一个长度为a.ndim(数组a的轴数)的元组
				nonzero(
					# clusterAssment第一列从矩阵转为数组，和j做==运算，得到一个布尔数组，也就是筛选属于j聚类的数据
					clusterAssment[:, 0].A == j
				)[0]
			]
			# 解析结束~~~
			
			# 这一个for循环的意义在于，循环k个聚类
			# 从原始数据矩阵中取出属于当前聚类的原始数据矩阵
			# 按列求出平均数，得到一个新的聚类中心
			# 替换掉原来的质心
			'''
			# pointsInCluster属于j聚类中心的原始数据
			pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]]
			# centroids聚类中心
			# 取出原始数据条目之后，按列计算均值
			centroids[j, :] = mean(pointsInCluster, axis=0)
		# 此时，如果clusterChanged为True，则while循环继续，一直循环下去，直到所有数据真正聚类，每个数据点所属的聚类中心不变
		# 本次聚类结束
	
	print('k_means函数执行结束，数据聚类完成！')
	# 返回(聚类中心点)和聚类之后的(所属聚类>>欧氏距离**2)矩阵，供绘图函数使用
	# 也就是返回了(聚类集)(所数据类和原始数据的位置对照表)
	return centroids, clusterAssment


# 显示您的聚类结果，只支持二维数据
def showCluster(dataSet, k, centroids, clusterAssment):
	# 获取数据集的行列数，列数也就是维数
	numSamples, dim = dataSet.shape
	# 判断如果不是二维，返回错误代号1
	if dim != 2:
		print "数据不是二维的，没法绘图！"
		return 1
	# 列出可使用的聚类数据点图形样式
	mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
	if k > len(mark):
		print('对不起，你定义的聚类中心太多，请修改')
		return 1
	# 绘图,画出所有样本点
	for i in xrange(numSamples):
		# 根据聚类中心的类型来选择对应的聚类数据点的样式类型
		markIndex = int(clusterAssment[i, 0])
		# 绘制一个点，根据数据的x、y、所属的聚类中心索引对应的样式
		plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])
	# 列出可使用的聚类中心点图形样式
	mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
	# 绘图,画出聚类中心
	for i in range(k):
		plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)
	plt.show()
