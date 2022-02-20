import torch

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


def generate_square_subsequent_mask(src_sz, tgt_sz):
	mask = (torch.tril(torch.ones((src_sz, tgt_sz))) == 1)
	return mask == False


if __name__ == "__main__":
	print(src_mask_attend_only_neighbour_tokens(10, 10 , 0))
