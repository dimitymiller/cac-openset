"""
	Randomly selects 5 trials of known and unknown classes for each dataset and saves for reference.
	1. MNIST, SVHN, CIFAR10 - 6 known, 4 unknown
	2. CIFAR+M - 4 known from non-animal subset of CIFAR10, M unknown from animal subset of CIFAR100
	3. TinyImageNet - 20 known, 180 unknown

	Dimity Miller, 2020
"""

import json
import random
import torchvision

random.seed(1000)

cifar100_animal = ['beaver', 'dolphin', 'otter', 'seal', 'whale',
'aquarium_fish', 'flatfish', 'ray', 'shark', 'trout',
'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
'bear', 'leopard', 'lion', 'tiger', 'wolf',
'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
'crab', 'lobster', 'snail', 'spider', 'worm',
'baby', 'boy', 'girl', 'man', 'woman',
'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel']

cifar10_nonanimal = ['plane', 'car', 'ship', 'truck']

def save_class_split(dataset, trial, classes):
	print("Saving {} trial {} to {}/class_splits/{}.json".format(dataset, trial, dataset, trial))
	file = open('{}/class_splits/{}.json'.format(dataset, trial), 'w')
	file.write(json.dumps(classes))
	file.close()



#MNIST, SVHN and CIFAR10 holds known classes for each dataset
num_unknown = 4
for dataset in ['MNIST', 'SVHN', 'CIFAR10']:
	for trial in range(5):
		classes = [i for i in range(10)]
		random.shuffle(classes)
		unknown_classes = classes[:num_unknown]
		known_classes = classes[num_unknown:]
		save_class_split(dataset, trial, {'Known': known_classes, 'Unknown': unknown_classes})

#CIFAR+M holds unknown animal classes to be used from CIFAR100
cifar100 = torchvision.datasets.CIFAR100('~/data')
mapping = cifar100.class_to_idx
for num_unknown in [10, 50]:
	for trial in range(5):
		unknown_classes = random.sample(cifar100_animal, num_unknown)
		unknown_idxs = [mapping[cl] for cl in unknown_classes]
		save_class_split('CIFAR+{}'.format(num_unknown), trial, {'Known': [0, 1, 8, 9], 'Unknown': unknown_idxs})

#TinyImageNet holds known classes for each dataset
num_unknown = 180
for trial in range(5):
	classes = [i for i in range(200)]
	random.shuffle(classes)
	unknown_classes = classes[:num_unknown]
	known_classes = classes[num_unknown:]
	save_class_split('TinyImageNet', trial, {'Known': known_classes, 'Unknown': unknown_classes})