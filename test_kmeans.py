# coding=utf-8
from numpy import *
import kmeans

# --------------------------
# step 1: load data
# 第一步：加载数据，就是根据文件中的数据创建一个矩阵
# 输出提示
print "第一步：加载数据"
# 创建一个列表，用来存储数据
dataSet = []
# 打开数据文件
fileIn = open('testSet.txt')
# 读取每一行，使用中间的tab进行切割
for line in fileIn.readlines():
	# 切割
	lineArr = line.strip().split('\t')
	# 存入数据列表(有序)，转化为float数据类型，数据文件的每行根据tab分割为两部分，使用下标来访问
	dataSet.append([float(lineArr[0]), float(lineArr[1])])
# --------------------------

# **************************
# step 2: clustering...
# 第二步：聚类...
print "第二步：聚类..."
# 使用mat函数把列表(数组)转换成矩阵
dataSet = mat(dataSet)
# 参数k为聚类中心数目
k = 4
# 执行kmeans文件中的k_means函数，传入矩阵和聚类中心数目k参数，分别赋值
centroids, clusterAssment = kmeans.k_means(dataSet, k)
# **************************

# ++++++++++++++++++++++++++
# step 3: show the result...
# 第三步：显示结果...
print "第三步：显示结果..."
# 执行kmeans文件中的显示图表函数
kmeans.showCluster(dataSet, k, centroids, clusterAssment)
# ++++++++++++++++++++++++++
