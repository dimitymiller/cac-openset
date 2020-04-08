"""
	Randomly select train and validation subsets from training datasets.
	80/20 split ratio used for all datasets except TinyImageNet, which will use 90/10.

	Dimity Miller, 2020
"""

import json
import random
import torchvision
import numpy as np

random.seed(1000)

def save_trainval_split(dataset, train_idxs, val_idxs):
	print("Saving {} Train/Val split to {}/trainval_idxs.json".format(dataset, dataset))
	file = open('{}/trainval_idxs.json'.format(dataset), 'w')
	file.write(json.dumps({'Train': train_idxs, 'Val': val_idxs}))
	file.close()

mnist = torchvision.datasets.MNIST('data')
svhn = torchvision.datasets.SVHN('data')
cifar10 = torchvision.datasets.CIFAR10('data')
tinyImagenet = torchvision.datasets.ImageFolder('data/tiny-imagenet-200/train')

datasets = {'MNIST': mnist, 'SVHN': svhn, 'CIFAR10': cifar10, 'TinyImageNet': tinyImagenet}
split = {'MNIST': 0.8, 'SVHN': 0.8, 'CIFAR10': 0.8, 'TinyImageNet': 0.9}

for datasetName in datasets.keys():
	dataset = datasets[datasetName]	

	#get class label for each image. svhn has different syntax as .labels
	try:
		targets = dataset.targets
		num_classes = len(dataset.classes)
	except:
		targets = dataset.labels
		num_classes = len(np.unique(targets))

	#save image idxs per class
	class_idxs = [[] for i in range(num_classes)]
	for i, lbl in enumerate(targets):
		class_idxs[lbl] += [i]

	#determine size of train subset
	class_size = [len(x) for x in class_idxs]
	class_train_size = [int(split[datasetName]*x) for x in class_size]

	#subset per class into train and val subsets randomly
	train_idxs = {}
	val_idxs = {}
	for class_num in range(num_classes):
		train_size = class_train_size[class_num]
		idxs = class_idxs[class_num]
		random.shuffle(idxs)
		train_idxs[class_num] = idxs[:train_size]
		val_idxs[class_num] = idxs[train_size:]

	save_trainval_split(datasetName, train_idxs, val_idxs)

	#cifar10 and cifar+m datasets can use the same training and val splits
	if 'CIFAR' in datasetName:
		save_trainval_split('CIFAR+10', train_idxs, val_idxs)
		save_trainval_split('CIFAR+50', train_idxs, val_idxs)