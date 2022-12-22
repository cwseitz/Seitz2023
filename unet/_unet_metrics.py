import torch
import numpy as np
import torch.nn.functional as F
from torch import nn

class MetricTools(object):
	def __init__(self):
		pass
	@staticmethod
	def _get_class_data(gt, pred, class_idx):
		#class_pred = pred[:, class_idx, :, :]
		#class_gt = gt[:, class_idx, :, :]
		class_pred = pred; class_gt = gt
		pred_flat = class_pred.contiguous().view(-1, )
		gt_flat = class_gt.contiguous().view(-1, )
		tp = torch.sum(gt_flat * pred_flat)
		fp = torch.sum(pred_flat) - tp
		fn = torch.sum(gt_flat) - tp

		tup = tp.item(), fp.item(), fn.item()

		return tup

def BackgroundPrecision(pred, gt, eps=1e-5):

	"""
	**Note: we need to implement a softmax layer to get valid probabilities
	The UNET model itself does not need this since CrossEntropyLoss operates on logits
	(non-normalized probabilities) by default.
	"""

	met = MetricTools()

	class_num = pred.size(1)
	activation_fn = nn.Softmax(dim=1)
	activated_pred = activation_fn(pred)
	activated_pred = torch.nn.functional.one_hot(activated_pred.argmax(dim=1), class_num).permute(0,3,1,2)
	tp,fp,fn = met._get_class_data(gt, activated_pred, 0)
	precision = tp/(tp+fp+eps)

	return precision

def InteriorPrecision(pred, gt, eps=1e-5):

	"""
	**Note: we need to implement a softmax layer to get valid probabilities
	The UNET model itself does not need this since CrossEntropyLoss operates on logits
	(non-normalized probabilities) by default.
	"""

	met = MetricTools()

	class_num = pred.size(1)
	activation_fn = nn.Softmax(dim=1)
	activated_pred = activation_fn(pred)
	activated_pred = torch.nn.functional.one_hot(activated_pred.argmax(dim=1), class_num).permute(0,3,1,2)
	tp,fp,fn = met._get_class_data(gt, activated_pred, 1)
	precision = tp/(tp+fp+eps)

	return precision

def BoundaryPrecision(pred, gt, eps=1e-5):

	"""
	**Note: we need to implement a softmax layer to get valid probabilities
	The UNET model itself does not need this since CrossEntropyLoss operates on logits
	(non-normalized probabilities) by default.
	"""

	met = MetricTools()

	class_num = pred.size(1)
	activation_fn = nn.Softmax(dim=1)
	activated_pred = activation_fn(pred)
	activated_pred = torch.nn.functional.one_hot(activated_pred.argmax(dim=1), class_num).permute(0,3,1,2)
	tp,fp,fn = met._get_class_data(gt, activated_pred, 2)
	recall = tp/(tp+fp+eps)

	return recall

def BackgroundRecall(pred, gt, eps=1e-5):

	"""
	**Note: we need to implement a softmax layer to get valid probabilities
	The UNET model itself does not need this since CrossEntropyLoss operates on logits
	(non-normalized probabilities) by default.
	"""

	met = MetricTools()

	class_num = pred.size(1)
	activation_fn = nn.Softmax(dim=1)
	activated_pred = activation_fn(pred)
	activated_pred = torch.nn.functional.one_hot(activated_pred.argmax(dim=1), class_num).permute(0,3,1,2)
	tp,fp,fn = met._get_class_data(gt, activated_pred, 0)
	recall = tp/(tp+fn+eps)

	return recall

def InteriorRecall(pred, gt, eps=1e-5):

	"""
	**Note: we need to implement a softmax layer to get valid probabilities
	The UNET model itself does not need this since CrossEntropyLoss operates on logits
	(non-normalized probabilities) by default.
	"""

	met = MetricTools()

	class_num = pred.size(1)
	activation_fn = nn.Softmax(dim=1)
	activated_pred = activation_fn(pred)
	activated_pred = torch.nn.functional.one_hot(activated_pred.argmax(dim=1), class_num).permute(0,3,1,2)
	tp,fp,fn = met._get_class_data(gt, activated_pred, 1)
	recall = tp/(tp+fn+eps)

	return recall

def BoundaryRecall(pred, gt, eps=1e-5):

	"""
	**Note: we need to implement a softmax layer to get valid probabilities
	The UNET model itself does not need this since CrossEntropyLoss operates on logits
	(non-normalized probabilities) by default.
	"""

	met = MetricTools()

	class_num = pred.size(1)
	activation_fn = nn.Softmax(dim=1)
	activated_pred = activation_fn(pred)
	activated_pred = torch.nn.functional.one_hot(activated_pred.argmax(dim=1), class_num).permute(0,3,1,2)
	tp,fp,fn = met._get_class_data(gt, activated_pred, 2)
	recall = tp/(tp+fn+eps)

	return recall
