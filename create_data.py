# coding=utf-8
import codecs
import random


def create_data(num):
	f_data = open('testSet2.txt', 'w')
	for i in xrange(num):
		a = random.randint(-100, 100)
		b = random.randint(-100, 100)
		f_data.write('{0}	{1}\n'.format(a, b))
	f_data.close()


create_data(100)
