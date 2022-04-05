import torch


def diagonal_mask(src_len, tgt_len, dist = 1):
	src_mask = torch.ones(size=(src_len, tgt_len))
	for i in range(src_len):
		for j in range(tgt_len):
			if abs(i - j) <= dist:
				src_mask[i][j] = 0
	
	return src_mask == 1

def src_mask_attend_only_neighbour_tokens(src_len, tgt_len, dist=2):
	"""
	len : is number of tokens
	range: number of neighbouring tokens to attend over.
	Purpose: create a source mask, such that attention for any token is calculate over its neighbouring tokens only.
	Except for first cls token, cls token attends to all the tokens
	value True means element is masked, False means not masked
	"""
	
	src_mask = torch.ones(size=(src_len, tgt_len))
	src_mask[0, :] = 0
	for i in range(src_len):
		for j in range(tgt_len):
			if abs(i - j) <= dist:
				src_mask[i][j] = 0
	
	return src_mask == 1


def generate_square_subsequent_mask(src_sz, tgt_sz , max_len  = 200):
	#mask = (torch.tril(torch.ones((src_sz, tgt_sz))) == 1)
	mask = torch.ones((src_sz, tgt_sz), dtype= torch.bool)
	for i in range(src_sz):
		for j in range(max(0, i-max_len), i+1):
			mask[i][j] = False
			
	return mask

def generate_sem_token_mask(src_sz, tgt_sz, len=6):
	mask = torch.ones((src_sz, tgt_sz), dtype= torch.bool)
	for i in range(0, src_sz):
		for j in range( int(i/len)*len , int(i/len)*len+int(i%len)+1):
			if j < tgt_sz:
				mask[i][j] =False
	return mask

def generate_token_mask(src_sz, tgt_sz,rng):
	len, rng= rng
	mask = torch.ones((src_sz, tgt_sz), dtype= torch.bool)
	for i in range(0, src_sz):
		for j in range(0, rng):
			mask[i][max(0, int(i/len)-j)] =False
			mask[i][min(tgt_sz-1,int(i / len) + j)] = False
	
	return mask


# mask used by models which make ABFN token based prediction
def generate_token_mask2(src_sz, tgt_sz, rng=(7,4)):
	len, rng = rng
	mask = torch.ones((src_sz, tgt_sz), dtype=torch.bool)
	for i in range(0, src_sz):
		k= int(i/len)*rng
		for j in range(rng):
			if (k+j)< tgt_sz:
				mask[i][k+j] =False
	return mask


if __name__ == "__main__":
	print(src_mask_attend_only_neighbour_tokens(21, 10 , 2))
