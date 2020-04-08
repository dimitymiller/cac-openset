"""
	Train a closed-set classifier on the datasets. 

	This training script has been adapted from https://github.com/kuangliu/pytorch-cifar

	Dimity Miller, 2020
"""
import torch
import torch.nn as nn
import torch.optim as optim

import json

import torchvision
import torchvision.transforms as tf

import argparse

import datasets.utils as dataHelper

from networks import closedSetClassifier

from utils import progress_bar

import os


parser = argparse.ArgumentParser(description='Closed Set Classifier Training')
parser.add_argument('--dataset', required = True, type = str, help='Dataset for training', 
									choices = ['MNIST', 'SVHN', 'CIFAR10', 'CIFAR+10', 'CIFAR+50', 'TinyImageNet'])
parser.add_argument('--trial', default = 0, type = int, help='Trial number, 0-4 is provided')
parser.add_argument('--resume', '-r', action='store_true', help='Resume from the checkpoint')
parser.add_argument('--tensorboard', '-t', action='store_true', help='Plot on tensorboardX')
parser.add_argument('--name', default = "", type = str, help='Optional name for saving and tensorboard') 
args = parser.parse_args()

if args.tensorboard:
	from tensorboardX import SummaryWriter

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#parameters useful when resuming and finetuning
best_acc = 0
start_epoch = 0

#Create dataloader for training
print('==> Preparing data..')
with open('datasets/config.json') as config_file:
	cfg = json.load(config_file)[args.dataset]

trainloader, valloader, _, mapping = dataHelper.get_train_loaders(args.dataset, args.trial, cfg)

print('==> Building network..')
net = closedSetClassifier.closedSetClassifier(cfg['num_known_classes'], cfg['im_channels'], cfg['im_size'],
												init_weights = not args.resume, dropout = cfg['dropout'])
net = net.to(device)

if args.resume:
	# Load checkpoint.
	print('==> Resuming from checkpoint..')
	assert os.path.isdir('networks/weights/{}'.format(args.dataset)), 'Error: no checkpoint directory found!'
	
	checkpoint = torch.load('networks/weights/{}/{}_{}_{}closedSetClassifier.pth'.format(args.dataset, args.dataset, args.trial, args.name))

	best_acc = checkpoint['acc']
	start_epoch = checkpoint['epoch']
	net.load_state_dict(checkpoint['net'])


criterion = nn.CrossEntropyLoss()
training_iter = int(args.resume)
optimizer = optim.SGD(net.parameters(), lr = cfg['closedset_training']['learning_rate'][training_iter], 
							momentum = 0.9, weight_decay = cfg['closedset_training']['weight_decay'])


if args.tensorboard:
	writer = SummaryWriter('runs/{}_{}_{}ClosedSet'.format(args.dataset, args.trial, args.name))

# Training
def train(epoch):
	print('\nEpoch: %d' % epoch)
	net.train()
	train_loss = 0
	correct = 0
	total = 0

	for batch_idx, (inputs, targets) in enumerate(trainloader):
		inputs, targets = inputs.to(device), targets.to(device)
		targets = torch.Tensor([mapping[x] for x in targets]).long().to(device)
		optimizer.zero_grad()

		outputs = net(inputs)
		loss = criterion(outputs, targets)

		if args.tensorboard and batch_idx%3 == 0:
			writer.add_scalar('train/CE_Loss', loss.item(), batch_idx + (epoch*len(trainloader)))

		loss.backward()

		optimizer.step()

		train_loss += loss.item()
		_, predicted = outputs.max(1)

		total += targets.size(0)
		correct += predicted.eq(targets).sum().item()

		progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
			% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
	
	if args.tensorboard:
		acc = 100.*correct/total
		writer.add_scalar('train/accuracy', acc, epoch)

def val(epoch):
	global best_acc
	net.eval()
	test_loss = 0
	correct = 0
	total = 0
	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(valloader):
			inputs = inputs.to(device)
			targets = torch.Tensor([mapping[x] for x in targets]).long().to(device)

			outputs = net(inputs)
			
			_, predicted = outputs.max(1)
			
			total += targets.size(0)

			correct += predicted.eq(targets).sum().item()

			progress_bar(batch_idx, len(valloader), 'Acc: %.3f%% (%d/%d)'
				% (100.*correct/total, correct, total))
   
	# Save checkpoint.
	acc = 100.*correct/total

	if acc > best_acc:
		print('Saving..')
		state = {
			'net': net.state_dict(),
			'acc': acc,
			'epoch': epoch,
		}
		if not os.path.isdir('networks/weights/{}'.format(args.dataset)):
			os.mkdir('networks/weights/{}'.format(args.dataset))
		torch.save(state, 'networks/weights/{}/{}_{}_{}closedSetClassifier.pth'.format(args.dataset, args.dataset, args.trial, args.name))
		best_acc = acc

	if args.tensorboard:
		writer.add_scalar('val/accuracy', acc, epoch)

max_epoch = cfg['closedset_training']['max_epoch'][training_iter]+start_epoch
for epoch in range(start_epoch, start_epoch+max_epoch):
	train(epoch)
	val(epoch)

