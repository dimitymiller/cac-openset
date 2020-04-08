"""
	Functions useful for creating experiment datasets and dataloaders.

	Dimity Miller, 2020
"""

import torch
import torchvision
import torchvision.transforms as tf
import json
from torch.autograd import Variable
import numpy as np
import random
random.seed(1000)

def get_train_loaders(datasetName, trial_num, cfg):
	"""
		Create training dataloaders.

		datasetName: name of dataset
		trial_num: trial number dictating known/unknown class split
		cfg: config file

		returns trainloader, evalloader, testloader, mapping - changes labels from original to known class label
	"""
	trainSet, valSet, testSet, _ = load_datasets(datasetName, cfg, trial_num)

	with open("datasets/{}/trainval_idxs.json".format(datasetName)) as f:
		trainValIdxs = json.load(f)
		train_idxs = trainValIdxs['Train']
		val_idxs = trainValIdxs['Val']

	with open("datasets/{}/class_splits/{}.json".format(datasetName, trial_num)) as f:
		class_splits = json.load(f)
		known_classes = class_splits['Known']

	trainSubset = create_dataSubsets(trainSet, known_classes, train_idxs)
	valSubset = create_dataSubsets(valSet, known_classes, val_idxs)
	testSubset = create_dataSubsets(testSet, known_classes)

	#create a mapping from dataset target class number to network known class number
	mapping = create_target_map(known_classes, cfg['num_classes'])

	batch_size = cfg['batch_size']

	trainloader = torch.utils.data.DataLoader(trainSubset, batch_size=batch_size, shuffle=True, num_workers = cfg['dataloader_workers'])
	valloader = torch.utils.data.DataLoader(valSubset, batch_size=batch_size, shuffle=True)
	testloader = torch.utils.data.DataLoader(testSubset, batch_size=batch_size, shuffle=True)

	return trainloader, valloader, testloader, mapping

def get_eval_loaders(datasetName, trial_num, cfg):
	"""
		Create evaluation dataloaders.

		datasetName: name of dataset
		trial_num: trial number dictating known/unknown class split
		cfg: config file

		returns knownloader, unknownloader, mapping - changes labels from original to known class label
	"""
	if '+' in datasetName or 'All' in datasetName:
		_, _, testSet, unknownSet = load_datasets(datasetName, cfg, trial_num)
	else:
		_, _, testSet, _ = load_datasets(datasetName, cfg, trial_num)

	with open("datasets/{}/class_splits/{}.json".format(datasetName, trial_num)) as f:
		class_splits = json.load(f)
		known_classes = class_splits['Known']
		unknown_classes = class_splits['Unknown']

	testSubset = create_dataSubsets(testSet, known_classes)
	
	if '+' in datasetName or 'All' in datasetName:
		unknownSubset = create_dataSubsets(unknownSet, unknown_classes)
	else:
		unknownSubset = create_dataSubsets(testSet, unknown_classes)

	#create a mapping from dataset target class number to network known class number
	mapping = create_target_map(known_classes, cfg['num_classes'])

	batch_size = cfg['batch_size']

	knownloader = torch.utils.data.DataLoader(testSubset, batch_size=batch_size, shuffle=False)
	unknownloader = torch.utils.data.DataLoader(unknownSubset, batch_size=batch_size, shuffle=False)

	return knownloader, unknownloader, mapping

def get_data_stats(dataset, known_classes):
	"""
		Calculates mean and std of data in a dataset.

		dataset: dataset to calculate mean and std of
		known_classes: what classes are known and should be included

		returns means and stds of data, across each colour channel
	"""
	try:
		ims = np.asarray(dataset.data)
		try:
			labels = np.asarray(dataset.targets)
		except:
			labels = np.asarray(dataset.labels)
		
		mask = labels == 1000
		for cl in known_classes:
			mask = mask | (labels == cl)
		known_ims = ims[mask]

		means = []
		stds = []
		if len(np.shape(known_ims)) < 4: 
			means += [known_ims.mean()/255]
			stds += [known_ims.std()/255]
		else:
			for i in range():
				means += [known_ims[:, :, :, i].mean()/255]
				stds += [known_ims[:, :, :, i].std()/255]
	except:
		imloader = torch.utils.data.DataLoader(dataset, batch_size = 128, shuffle = False)

		r_data = []
		g_data = []
		b_data = []
		for i, data in enumerate(imloader):
			im, labels = data
			mask = labels == 1000
			for cl in known_classes:
				mask = mask | (labels == cl)			
			if torch.sum(mask) == 0:
				continue
			im = im[mask]
			r_data += im[:, 0].detach().tolist()
			g_data += im[:, 1].detach().tolist()
			b_data += im[:, 2].detach().tolist()
		means = [np.mean(r_data), np.mean(g_data), np.mean(b_data)]
		stds = [np.std(r_data), np.std(g_data), np.std(b_data)]
	return means, stds

def load_datasets(datasetName, cfg, trial_num):
	"""
		Load all datasets for training/evaluation.

		datasetName: name of dataset
		cfg: config file
		trial_num: trial number dictating known/unknown class split

		returns trainset, valset, knownset, unknownset
	"""
	with open("datasets/{}/class_splits/{}.json".format(datasetName, trial_num)) as f:
		class_splits = json.load(f)
		known_classes = class_splits['Known']

	#controls data transforms
	means = cfg['data_mean'][trial_num]
	stds = cfg['data_std'][trial_num]
	flip = cfg['data_transforms']['flip']
	rotate = cfg['data_transforms']['rotate']
	scale_min = cfg['data_transforms']['scale_min']

	transforms = {
	    'train': tf.Compose([
	        tf.Resize(cfg['im_size']),
	        tf.RandomResizedCrop(cfg['im_size'], scale = (scale_min, 1.0)),
	        tf.RandomHorizontalFlip(flip),
	        tf.RandomRotation(rotate),
	        tf.ToTensor(),
	        tf.Normalize(means, stds)
	    ]),
	    'val': tf.Compose([
	        tf.Resize(cfg['im_size']),
	        tf.ToTensor(),
	        tf.Normalize(means, stds)
	    ]),
	    'test': tf.Compose([
	        tf.Resize(cfg['im_size']),
	        tf.ToTensor(),
	        tf.Normalize(means, stds)
	    ])
	}

	unknownSet = None
	if datasetName == "MNIST":
		trainSet = torchvision.datasets.MNIST('datasets/data', transform = transforms['train'], download = True)
		valSet = torchvision.datasets.MNIST('datasets/data', transform = transforms['val'])
		testSet = torchvision.datasets.MNIST('datasets/data', train = False, transform = transforms['test'])
	elif "CIFAR" in datasetName:
		trainSet = torchvision.datasets.CIFAR10('datasets/data', transform = transforms['train'], download = True)
		valSet = torchvision.datasets.CIFAR10('datasets/data', transform = transforms['val'])
		testSet = torchvision.datasets.CIFAR10('datasets/data', train = False, transform = transforms['test'])
		if '+' in datasetName:
			unknownSet = torchvision.datasets.CIFAR100('datasets/data', train = False, transform = transforms['test'], download = True)
	elif datasetName == "SVHN":
		trainSet = torchvision.datasets.SVHN('datasets/data', transform = transforms['train'], download = True)
		valSet = torchvision.datasets.SVHN('datasets/data', transform = transforms['val'])
		testSet = torchvision.datasets.SVHN('datasets/data', split = 'test', transform = transforms['test'])
	elif datasetName == "TinyImageNet":
		trainSet = torchvision.datasets.ImageFolder('datasets/data/tiny-imagenet-200/train', transform = transforms['train'])
		valSet = torchvision.datasets.ImageFolder('datasets/data/tiny-imagenet-200/train', transform = transforms['val'])
		testSet = torchvision.datasets.ImageFolder('datasets/data/tiny-imagenet-200/val', transform = transforms['test'])
	else:
		print("Sorry, that dataset has not been implemented.")
		exit()

	return trainSet, valSet, testSet, unknownSet

def get_anchor_loaders(datasetName, trial_num, cfg):
	"""
		Supply trainloaders, with no extra rotate/crop data augmentation, for calculating anchor class centres.

		datasetName: name of dataset
		trial_num: trial number dictating known/unknown class split
		cfg: config file

		returns trainloader and trainloaderFlipped (horizontally) 
	"""
	trainSet, trainSetFlipped = load_anchor_datasets(datasetName, cfg, trial_num)

	with open("datasets/{}/trainval_idxs.json".format(datasetName)) as f:
		trainValIdxs = json.load(f)
		train_idxs = trainValIdxs['Train']
		val_idxs = trainValIdxs['Val']

	with open("datasets/{}/class_splits/{}.json".format(datasetName, trial_num)) as f:
		class_splits = json.load(f)
		known_classes = class_splits['Known']

	trainSubset = create_dataSubsets(trainSet, known_classes, train_idxs)
	trainSubsetFlipped = create_dataSubsets(trainSetFlipped, known_classes, train_idxs)

	trainloader = torch.utils.data.DataLoader(trainSubset, batch_size=128)
	trainloaderFlipped = torch.utils.data.DataLoader(trainSubsetFlipped, batch_size=128)

	return trainloader, trainloaderFlipped

def load_anchor_datasets(datasetName, cfg, trial_num):
	"""
		Load train datasets, with no extra rotate/crop data augmentation, for calculating anchor class centres.

		datasetName: name of dataset
		cfg: config file
		trial_num: trial number dictating known/unknown class split

		returns trainset and trainsetFlipped (horizontally) 
	"""
	with open("datasets/{}/class_splits/{}.json".format(datasetName, trial_num)) as f:
		class_splits = json.load(f)
		known_classes = class_splits['Known']

	means = cfg['data_mean'][trial_num]
	stds = cfg['data_std'][trial_num]

	#for digit datasets, we don't want to provide a flip dataset
	if datasetName == "MNIST" or datasetName == "SVHN":
		flip = 0
	else:
		flip = 1

	transforms = {
		'train': tf.Compose([
			tf.Resize(cfg['im_size']), tf.CenterCrop(cfg['im_size']),
			tf.ToTensor(),
			tf.Normalize(means, stds)
		]),
		'trainFlip': tf.Compose([
			tf.Resize(cfg['im_size']), tf.CenterCrop(cfg['im_size']),
			tf.RandomHorizontalFlip(flip),
			tf.ToTensor(),
			tf.Normalize(means, stds)
		])
	}
	if datasetName == "MNIST":
		trainSet = torchvision.datasets.MNIST('datasets/data', transform = transforms['train'])
		trainSetFlip = torchvision.datasets.MNIST('datasets/data', transform = transforms['trainFlip'])
	elif "CIFAR" in datasetName:
		trainSet = torchvision.datasets.CIFAR10('datasets/data', transform = transforms['train'])
		trainSetFlip = torchvision.datasets.CIFAR10('datasets/data', transform = transforms['trainFlip'])
	elif datasetName == "SVHN":
		trainSet = torchvision.datasets.SVHN('datasets/data', transform = transforms['train'])
		trainSetFlip = torchvision.datasets.SVHN('datasets/data', transform = transforms['trainFlip'])
	elif datasetName == "TinyImageNet":
		trainSet = torchvision.datasets.ImageFolder('datasets/data/tiny-imagenet-200/train', transform = transforms['train'])
		trainSetFlip = torchvision.datasets.ImageFolder('datasets/data/tiny-imagenet-200/train', transform = transforms['trainFlip'])
	else:
		print("Sorry, that dataset has not been implemented.")
		exit()

	return trainSet, trainSetFlip

def create_dataSubsets(dataset, classes_to_use, idxs_to_use = None):
	"""
		Returns dataset subset that satisfies class and idx restraints.
		dataset: torchvision dataset
		classes_to_use: classes that are allowed in the subset (known vs unknown)
		idxs_to_use: image indexes that are allowed in the subset (train vs val, not relevant for test)

		returns torch Subset
	"""
	import torch

	#get class label for dataset images. svhn has different syntax as .labels
	try:
		targets = dataset.targets
	except:
		targets = dataset.labels

	subset_idxs = []
	if idxs_to_use == None:
		for i, lbl in enumerate(targets):
			if lbl in classes_to_use:
				subset_idxs += [i]
	else:
		for class_num in idxs_to_use.keys():
			if int(class_num) in classes_to_use:
				subset_idxs += idxs_to_use[class_num]

	dataSubset = torch.utils.data.Subset(dataset, subset_idxs)
	return dataSubset

def create_target_map(known_classes, num_classes):
	"""
		Creates a mapping from original dataset labels to new 'known class' training label
		known_classes: classes that will be trained with
		num_classes: number of classes the dataset typically has
		
		returns mapping - a dictionary where mapping[original_class_label] = known_class_label
	"""
	mapping = [None for i in range(num_classes)]
	
	known_classes.sort()
	for i, num in enumerate(known_classes):
		mapping[num] = i

	return mapping