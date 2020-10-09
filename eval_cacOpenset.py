"""
	Evaluate average performance for our proposed CAC open-set classifier on a given dataset.

	Dimity Miller, 2020
"""


import argparse
import json

import torchvision
import torchvision.transforms as tf
import torch
import torch.nn as nn

from networks import openSetClassifier
import datasets.utils as dataHelper
from utils import find_anchor_means, gather_outputs

import metrics
import scipy.stats as st
import numpy as np

parser = argparse.ArgumentParser(description='Closed Set Classifier Training')
parser.add_argument('--dataset', default = "MNIST", type = str, help='Dataset for evaluation', 
									choices = ['MNIST', 'SVHN', 'CIFAR10', 'CIFAR+10', 'CIFAR+50', 'CIFARAll', 'TinyImageNet'])
parser.add_argument('--num_trials', default = 5, type = int, help='Number of trials to average results over?')
parser.add_argument('--start_trial', default = 0, type = int, help='Trial number to start evaluation for?')
parser.add_argument('--name', default = '', type = str, help='Name of training script?')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

all_accuracy = []
all_auroc = []

for trial_num in range(args.start_trial, args.start_trial+args.num_trials):
	print('==> Preparing data for trial {}..'.format(trial_num))
	with open('datasets/config.json') as config_file:
		cfg = json.load(config_file)[args.dataset]

	#Create dataloaders for evaluation
	knownloader, unknownloader, mapping = dataHelper.get_eval_loaders(args.dataset, trial_num, cfg)

	print('==> Building open set network for trial {}..'.format(trial_num))
	net = openSetClassifier.openSetClassifier(cfg['num_known_classes'], cfg['im_channels'], cfg['im_size'], dropout = cfg['dropout'])
	checkpoint = torch.load('networks/weights/{}/{}_{}_{}CACclassifierAnchorLoss.pth'.format(args.dataset, args.dataset, trial_num, args.name))

	net = net.to(device)
	net_dict = net.state_dict()
	pretrained_dict = {k: v for k, v in checkpoint['net'].items() if k in net_dict}
	if 'anchors' not in pretrained_dict.keys():
		pretrained_dict['anchors'] = checkpoint['net']['means']
	net.load_state_dict(pretrained_dict)
	net.eval()

	#find mean anchors for each class
	anchor_means = find_anchor_means(net, mapping, args.dataset, trial_num, cfg, only_correct = True)
	net.set_anchors(torch.Tensor(anchor_means))

	
	print('==> Evaluating open set network accuracy for trial {}..'.format(trial_num))
	x, y = gather_outputs(net, mapping, knownloader, data_idx = 1, calculate_scores = True)
	accuracy = metrics.accuracy(x, y)
	all_accuracy += [accuracy]

	print('==> Evaluating open set network AUROC for trial {}..'.format(trial_num))
	xK, yK = gather_outputs(net, mapping, knownloader, data_idx = 1, calculate_scores = True)
	xU, yU = gather_outputs(net, mapping, unknownloader, data_idx = 1, calculate_scores = True, unknown = True)

	auroc = metrics.auroc(xK, xU)
	all_auroc += [auroc]

mean_auroc = np.mean(all_auroc)
mean_acc = np.mean(all_accuracy)

print('Raw Top-1 Accuracy: {}'.format(all_accuracy))
print('Raw AUROC: {}'.format(all_auroc))
print('Average Top-1 Accuracy: {}'.format(mean_acc))
print('Average AUROC: {}'.format(mean_auroc))