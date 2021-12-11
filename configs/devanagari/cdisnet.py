#language specific changes:
character = 'ऀँंःऄअआइईउऊऋऌऍऎएऐऑऒओऔकखगघङचछजझञटठडढणतथदधनऩपफबभमयरऱलळऴवशषसहऺऻ़ऽािीुूृॄॅॆेैॉॊोौ्ॎॏॐ॒॑॓॔ॕॖॗक़ख़ग़ज़ड़ढ़फ़य़ॠॡॢॣ।॥०१२३४५६७८९%/?:,.-'
test_sensitive = False
test_character = 'ऀँंःऄअआइईउऊऋऌऍऎएऐऑऒओऔकखगघङचछजझञटठडढणतथदधनऩपफबभमयरऱलळऴवशषसहऺऻ़ऽािीुूृॄॅॆेैॉॊोौ्ॎॏॐ॒॑॓॔ॕॖॗक़ख़ग़ज़ड़ढ़फ़य़ॠॡॢॣ।॥०१२३४५६७८९%/?:,.-'
batch_max_length = 35
test_folder_names = ['IIIT']  ###
data_root = '/usr/datasets/synthetic_text_dataset/lmdb_dataset_Hindi/hindi/'

# work directory
root_workdir = 'workdir'
# sample_per_gpu
samples_per_gpu = 64
###############################################################################
# 1. inference
size = (32, 100)
mean, std = 0.5, 0.5

sensitive = True

dropout = 0.1
n_e = 9
n_d = 3
hidden_dim = 256
n_head = 8
batch_norm = dict(type='BN')
layer_norm = dict(type='LayerNorm', normalized_shape=hidden_dim)
num_class = len(character) + 2

inference = dict(
	transform=[
		dict(type='Sensitive', sensitive=sensitive),
		dict(type='Filter', need_character=character),
		dict(type='ToGray'),
		dict(type='Resize', size=size),
		dict(type='Normalize', mean=mean, std=std),
		dict(type='ToTensor'),
	],
	converter=dict(
		type='AttnConverter',
		character=character,
		batch_max_length=batch_max_length,
		go_last=True,
	),
	model=dict(
		type='Cdisnet',
		flags="/home/shubham/Documents/MTP/text-recognition-models/vedastr/configs/devanagari/config/cdistnet.yml",
		num_class = len(character)+2,
		need_text=True
	),
	postprocess=dict(
		sensitive=test_sensitive,
		character=test_character,
	),
)

###############################################################################
# 2.common

common = dict(
	seed=1111,
	logger=dict(
		handlers=(
			dict(type='StreamHandler', level='INFO'),
			dict(type='FileHandler', level='INFO'),
		),
	),
	cudnn_deterministic=False,
	cudnn_benchmark=True,
	metric=dict(type='Accuracy'),
	dist_params=dict(backend='nccl'),
)

###############################################################################
dataset_params = dict(
	batch_max_length=batch_max_length,
	data_filter=False,
	character=character,
)
test_dataset_params = dict(
	batch_max_length=batch_max_length,
	data_filter=False,
	character=test_character,
)

# data_root = './data/data_lmdb_release/'
###############################################################################
# 3. test
test_root = data_root + 'evaluation/'
# test_folder_names = ['CUTE80', 'IC03_867', 'IC13_1015', 'IC15_2077',
#                     'IIIT5k_3000', 'SVT', 'SVTP']


test_dataset = [dict(type='LmdbDataset', root=test_root + f_name,
                     **test_dataset_params) for f_name in test_folder_names]

test = dict(
	data=dict(
		dataloader=dict(
			type='DataLoader',
			samples_per_gpu=samples_per_gpu,
			workers_per_gpu=4,
			shuffle=False,
		),
		sampler=dict(type='DefaultSampler', shuffle=False),
		dataset=test_dataset,
		transform=[
			dict(type='Sensitive', sensitive=test_sensitive),
			dict(type='Filter', need_character=test_character),
			dict(type='ToGray'),
			dict(type='Resize', size=size),
			dict(type='Normalize', mean=mean, std=std),
			dict(type='ToTensor'),
		],
	),
	postprocess_cfg=dict(
		sensitive=test_sensitive,
		character=test_character,
	),
)

###############################################################################
# 4. train
## MJ dataset
train_root_mj = data_root + 'training/MJ/'
mj_folder_names = ['MJ_test', 'MJ_valid', 'MJ_train']
## ST dataset
train_root_st = data_root + 'training/ST/'

train_dataset_mj = [dict(type='LmdbDataset', root=train_root_mj + folder_name)
                    for folder_name in mj_folder_names]
train_dataset_st = [dict(type='LmdbDataset', root=train_root_st)]

# valid
valid_root = data_root + 'validation/'
valid_dataset = dict(type='LmdbDataset', root=valid_root, **test_dataset_params)

train_transforms = [
	dict(type='Sensitive', sensitive=sensitive),
	dict(type='Filter', need_character=character),
	dict(type='ToGray'),
	dict(type='ExpandRotate', limit=34, p=0.5),
	dict(type='Resize', size=size),
	dict(type='Normalize', mean=mean, std=std),
	dict(type='ToTensor'),
]

max_epochs = 6
milestones = [2, 4]  # epoch start from 0, so 2 means lr decay at 3 epoch, 4 means lr decay at the end of

train = dict(
	data=dict(
		train=dict(
			dataloader=dict(
				type='DataLoader',
				samples_per_gpu=samples_per_gpu,
				workers_per_gpu=4,
			),
			sampler=dict(
				type='BalanceSampler',
				samples_per_gpu=samples_per_gpu,
				shuffle=True,
				oversample=True,
				seed=common['seed'],
			),
			dataset=dict(
				type='ConcatDatasets',
				datasets=[
					dict(
						type='ConcatDatasets',
						datasets=train_dataset_mj,
					),
					dict(
						type='ConcatDatasets',
						datasets=train_dataset_st,
					),
				],
				batch_ratio=[0.5, 0.5],
				**dataset_params,
			),
			transform=train_transforms,
		),
		val=dict(
			dataloader=dict(
				type='DataLoader',
				samples_per_gpu=samples_per_gpu,
				workers_per_gpu=4,
				shuffle=False,
			),
			dataset=valid_dataset,
			transform=test['data']['transform'],
		),
	),
	optimizer=dict(type='Adam', lr=3e-4),
	criterion=dict(type='CrossEntropyLoss'),
	lr_scheduler=dict(type='CosineLR',
	                  iter_based=True,
	                  warmup_epochs=0.1,
	                  ),
	max_epochs=max_epochs,
	log_interval=10,
	trainval_ratio=2000,
	snapshot_interval=20000,
	save_best=True,
	resume=None,
)
