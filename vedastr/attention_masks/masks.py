import torch

#TODO masking functions are not correct.

def src_mask_attend_only_neighbour_tokens(len,dist=1):
	"""
	len : is number of tokens
	range: number of neighbouring tokens to attend over.
	Purpose: create a source mask, such that attention for any token is calculate over its neighbouring tokens only.
	Except for first cls token, cls token attends to all the tokens
	"""
	
	src_mask= torch.zeros(size=(len, len))
	src_mask[0,:] =1
	for i in range(len):
		for j in range(len):
			if abs(i-j)<= dist:
				src_mask[i][j]=1
				
	return src_mask

def memory_mask(src_len, tgt_len, forward_range=1):
	"""
	for decoder should attend only those patches
	"""
	memory_mask = torch.zeros(tgt_len,src_len)
	memory_mask[:, 0] =1
	
	
	for i in range(tgt_len):
		memory_mask[i][0:min(i+forward_range, src_len)] =1
		
	return memory_mask
	
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def tgt_mask_full(tgt_len):
	"""
	dont compute attention over any tgt token, compute attention only over src
	"""
	return torch.zeros(tgt_len,tgt_len)
