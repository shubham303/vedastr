# work directory
import torch

root_workdir = 'workdir'
# sample_per_gpu
samples_per_gpu = 64
###############################################################################
# 1. inference
size = (224, 224)  # Note for DPT image size 224*224, but text image in general of form height <<
# width, ex . 32*224 need to change code according to that.

mean, std = 0.5, 0.5

character = '0123456789abcdefghijklmnopqrstuvwxyz'
sensitive = False
batch_max_length = 25

test_sensitive = False
test_character = '0123456789abcdefghijklmnopqrstuvwxyz'

num_class = len(character) + 2  # Attention based need two more characters: '[G0]' and '[S]'

dropout = 0.1

img_size = 224
patch_sizes = [4, 2, 2, 2]
embed_dims = [64, 128, 320, 512]
in_sizes = [img_size // 2 ** (i + 1) for i in range(0, 4)]
n_heads = [1, 2, 4, 8]
encoder_layers = [3, 4, 6, 3]
in_channels = [3 if i == 0 else embed_dims[i - 1] for i in range(4)]
Depatch = [False, True, True, True]
mlp_ratios = [8, 8, 4, 4]


fiducial_num = 20
hidden_dim = 512
norm_cfg = dict(type='BN')



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
		type='GModel',
		need_text=False,
		body=dict(
			type="GBody",
			pipelines=[
				dict(
					type="SequenceEncoderComponent",
					from_layer="input",
					to_layer="src",
					arch=dict(
						type="DPT",
						layers=[
							dict(
								type="TransformerEncoder",
								embedding=dict(
									type="PatchEmbed",
									img_size=in_sizes[i],
									patch_size=patch_sizes[i],
									in_chans=in_channels[i],
									embed_dim=embed_dims[i]
								) if not Depatch[i] else dict(
									type="Simple_DePatch",
									box_coder=dict(
										type="PointwhCoder",
										input_size=in_sizes[i],
										patch_count=in_sizes[i] / patch_sizes[i],
										weights=(1., 1., 1., 1.),
										pts=3,
										tanh=True,
										wh_bias=torch.tensor(5. / 3.).sqrt().log()
									),
									img_size=in_sizes[i],
									patch_size=patch_sizes[i],
									patch_pixels=3,
									patch_count=in_sizes[i] // patch_sizes[i],
									in_chans=in_channels[i],
									embed_dims=embed_dims[i],
									another_linear=True,
									use_GE=True,
									with_norm=True
								),
								position_encoder=dict(
									type='PositionEncoder1D',
									in_channels=embed_dims[i],
									max_len=100,
									dropout=dropout
								),
								encoder_layer=dict(
									type='TransformerEncoderLayer1D',
									attention=dict(
										type='MultiHeadAttention',
										in_channels=embed_dims[i],
										k_channels=embed_dims[i] // n_heads[i],
										v_channels=embed_dims[i] // n_heads[i],
										n_head=n_heads[i],
										dropout=dropout,
									),
									attention_norm=dict(type='LayerNorm', normalized_shape=embed_dims[i]),
									feedforward=dict(
										type='Feedforward',
										layers=[
											dict(type='FCModule', in_channels=embed_dims[i],
											     out_channels=embed_dims[i] *
											                  mlp_ratios[i],
											     bias=True,
											     activation='gelu', dropout=dropout),
											dict(type='FCModule', in_channels=embed_dims[i] * 4,
											     out_channels=embed_dims[i],
											     bias=True,
											     activation=None, dropout=dropout),
										],
									),
									feedforward_norm=dict(type='LayerNorm', normalized_shape=embed_dims[i]),
								),
								num_layers=encoder_layers[i]
							) for i in range(0, 4)
						],
						last_embedding_dim=embed_dims[-1],
						norm=dict(type='LayerNorm', normalized_shape=embed_dims[-1])
					)
				)
			]
		),
		head=dict(
			type='VisionTransformerFChead',
			in_channels=embed_dims[-1],
			num_class=num_class,
			from_layer='src',
			batch_max_length=batch_max_length
		),
		
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
	data_filter=True,
	character=character,
)
test_dataset_params = dict(
	batch_max_length=batch_max_length,
	data_filter=False,
	character=test_character,
)

# data_root = './data/data_lmdb_release/'
data_root = '/home/shubham/Documents/MTP/datasets/lmdb_datasets/'
###############################################################################
# 3. test
test_root = data_root + 'evaluation/'
# test_folder_names = ['CUTE80', 'IC03_867', 'IC13_1015', 'IC15_2077',
#                     'IIIT5k_3000', 'SVT', 'SVTP']


test_folder_names = ['SVT']
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
