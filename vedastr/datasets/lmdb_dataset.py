# modify from clovaai

import logging

import lmdb
import numpy as np
import six
from PIL import Image

from .base import BaseDataset
from .registry import DATASETS

logger = logging.getLogger()


@DATASETS.register_module
class LmdbDataset(BaseDataset):
	""" Read the data of lmdb format.
	Please refer to https://github.com/Media-Smart/vedastr/issues/27#issuecomment-691793593  # noqa 501
	if you have problems with creating lmdb format file.

	"""
	
	def __init__(self,
	             root: str,
	             transform=None,
	             character: str = 'abcdefghijklmnopqrstuvwxyz0123456789',
	             batch_max_length: int = 100000,
	             data_filter: bool = True):
		self.index_list = []
		self.is_valid_index =[]
		super(LmdbDataset, self).__init__(
			root=root,
			transform=transform,
			character=character,
			batch_max_length=batch_max_length,
			data_filter=data_filter
		)
	
	def get_name_list(self):
		self.env = lmdb.open(
			self.root,
			max_readers=32,
			readonly=True,
			lock=False,
			readahead=False,
			meminit=False)
		with self.env.begin(write=False) as txn:
			n_samples = int(txn.get('num-samples'.encode()))
			n_samples = min(4000000, n_samples)
		
			self.index_list = range(n_samples)
			# checking if (img, label) pair is valid or not is expensive operation. is_valid_index stores that
			# inforation once caluculated to reduce computation. -1 : not yet checked  0: data item is invalid  1 :
			# data item is valid
			self.is_valid_index = [-1]*n_samples
			self.samples = len(self.index_list)
	
	def read_data(self, index , txn):
		assert index <= len(self), 'index range error'
		index = self.index_list[index]

		# read next data item if data item is not valid.
		
		if self.is_valid_index[index] == 0 :
			return self.read_data((index + 1) % self.samples, txn)
		
		label_key = 'label-%09d'.encode() % index
		label = txn.get(label_key)
		
		if label is None or len(label) ==0 :
			self.is_valid_index[index] = 0
			return self.read_data((index + 1) % self.samples, txn)
		
		label = label.decode('utf-8')
		
		if self.is_valid_index[index] ==-1:
			if self.filter(label):
				self.is_valid_index[index] =0
				return self.read_data((index + 1) % self.samples, txn)
			else:
				self.is_valid_index[index] =1
			
		img_key = 'image-%09d'.encode() % index
		imgbuf = txn.get(img_key)
		
		buf = six.BytesIO()
		buf.write(imgbuf)
		buf.seek(0)
		img = Image.open(buf).convert('RGB')  # for color image
		img = np.array(img)
		
		return img, label
	
	def __getitem__(self, index):
		with self.env.begin(write=False) as txn:
			img, label = self.read_data(index , txn)
			if self.transforms:
				aug = self.transforms(image=img, label=label)
				img, label = aug['image'], aug['label']
			
			return img, label
