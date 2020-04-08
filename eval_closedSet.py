"""
	Evaluate average performance for a standard closed set classifier on a given dataset.

	Dimity Miller, 2020
"""


import argparse
import json

import torchvision
import torchvision.transforms as tf
import torch
import torch.nn as nn

from networks import closedSetClassifier
import datasets.utils as dataHelper

from sklearn import metrics as skmetrics
import metrics
import scipy.stats as st
import numpy as np

parser = argparse.ArgumentParser(description='Closed Set Classifier Training')
parser.add_argument('--dataset', default = "MNIST", type = str, help='Dataset for evaluation', 
									choices = ['MNIST', 'SVHN', 'CIFAR10', 'CIFAR+10', 'CIFAR+50', 'CIFARAll', 'TinyImageNet'])
parser.add_argument('--num_trials', default = 5, type = int, help='Number of trials to average results over')
parser.add_argument('--start_trial', default = 0, type = int, help='Trial number to start evaluation for?')
parser.add_argument('--name', default = '', type = str, help='What iteration of gaussian fitting in open set training?')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

all_accuracy = []
all_auroc = []
for trial_num in range(args.start_trial, args.start_trial+args.num_trials):
	print('==> Preparing data for trial {}..'.format(trial_num))
	with open('datasets/config.json') as config_file:
		cfg = json.load(config_file)[args.dataset]

	#Create dataloaders for training
	knownloader, unknownloader, mapping = dataHelper.get_eval_loaders(args.dataset, trial_num, cfg)

	###################Closed Set Network Evaluation##################################################################
	print('==> Building open set network for trial {}..'.format(trial_num))
	net = closedSetClassifier.closedSetClassifier(cfg['num_known_classes'], cfg['im_channels'], cfg['im_size'])
	checkpoint = torch.load('networks/weights/{}/{}_{}_{}closedSetClassifier.pth'.format(args.dataset, args.dataset, trial_num, args.name))

	net = net.to(device)
	net.load_state_dict(checkpoint['net'])
	net.eval()

	X = []
	y = []


	softmax = torch.nn.Softmax(dim = 1)
	for i, data in enumerate(knownloader):
		images, labels = data
		targets = torch.Tensor([mapping[x] for x in labels]).long().cuda()
		
		images = images.cuda()
		logits = net(images)
		scores = softmax(logits)

		X += scores.cpu().detach().tolist()
		y += targets.cpu().tolist()

	X = -np.asarray(X)
	y = np.asarray(y)

	accuracy = metrics.accuracy(X, y)
	all_accuracy += [accuracy]


	XU = []
	for i, data in enumerate(unknownloader):
		images, labels = data
		
		images = images.cuda()
		logits = net(images)
		scores = softmax(logits)
		XU += scores.cpu().detach().tolist()

	XU = -np.asarray(XU)
	auroc = metrics.auroc(X, XU)
	all_auroc += [auroc]

mean_acc = np.mean(all_accuracy)
mean_auroc = np.mean(all_auroc)

print('Raw Top-1 Accuracy: {}'.format(all_accuracy))
print('Raw AUROC: {}'.format(all_auroc))
print('Average Top-1 Accuracy: {}'.format(mean_acc))
print('Average AUROC: {}'.format(mean_auroc))