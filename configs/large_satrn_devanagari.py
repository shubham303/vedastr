#language specific changes:
character = 'ऀँंःऄअआइईउऊऋऌऍऎएऐऑऒओऔकखगघङचछजझञटठडढणतथदधनऩपफबभमयरऱलळऴवशषसहऺऻ़ऽािीुूृॄॅॆेैॉॊोौ्ॎॏॐ॒॑॓॔ॕॖॗॠ०१२३४५६७८९ॲ%/?:,.-'
test_sensitive = False
test_character = 'ऀँंःऄअआइईउऊऋऌऍऎएऐऑऒओऔकखगघङचछजझञटठडढणतथदधनऩपफबभमयरऱलळऴवशषसहऺऻ़ऽािीुूृॄॅॆेैॉॊोौ्ॎॏॐ॒॑॓॔ॕॖॗॠ०१२३४५६७८९ॲ'
batch_max_length = 35
test_folder_names = ['IIIT']  ###
#data_root = '/usr/datasets/synthetic_text_dataset/lmdb_dataset_Hindi/hindi/'
data_root = '/home/ocr/datasets/recognition/hindi/'
validation_folder_names=['MJ_valid', "ST_valid"]
mj_folder_names = ['MJ_test', 'MJ_train']

m = "ऀ  ँ ं ः  ॕ "
V = "ऄ ई ऊ ऍ  ऎ ऐ ऑ ऒ ओ औ"
CH = "अ आ उ ए इ ऌ क  ख  ग ऋ  घ  ङ  च  छ  ज  झ  ञ  ट  ठ  ड  ढ  ण  त  थ  द  ध  न  ऩ  प  फ  ब  भ  म  य  र  ऱ  ल  ळ  ऴ  व  " \
     "श  ष  " \
     "स  ह ॐ क़  ख़  ग़  ज़  ड़  ढ़  फ़  य़  ॠ  ॡ"
v = "ा  ि  ी  ु  ू  ृ  ॄ  ॉ  ॊ  ो  ौ  ॎ  ॏ ॑  ॒  ॓ ़ ॔  ॅ े ै ॆ ्  ॖ   ॗ ॢ  ॣ"
symbols = "।  ॥  ०  १  २  ३  ४  ५  ६  ७  ८  ९ %  /  ?  :  ,  .  -"


# language specific chanage end here.

# work directory
root_workdir = 'workdir'
# sample_per_gpu
samples_per_gpu = 32
###############################################################################
# 1. inference
size = (32, 100)
mean, std = 0.5, 0.5
sensitive = True
dropout = 0.1
n_e = 12
n_d = 6
hidden_dim = 512
n_head = 8
batch_norm = dict(type='BN')
layer_norm = dict(type='LayerNorm', normalized_shape=hidden_dim)
num_class = len(character) + 1
num_steps = batch_max_length + 1

inference = dict(
	transform=[
		dict(type='Sensitive', sensitive=sensitive),
		dict(type='Filter', need_character=character),
		dict(type='ToGray'),
		dict(type='Resize', size=size),
		dict(type='Normalize', mean=mean, std=std),
		#dict(type='Rotate'),
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
		need_text=True,
		body=dict(
			type='GBody',
			pipelines=[
				dict(
					type='FeatureExtractorComponent',
					from_layer='input',
					to_layer='cnn_feat',
					arch=dict(
						encoder=dict(
							backbone=dict(
								type='GBackbone',
								layers=[
									dict(type='ConvModule', in_channels=1, out_channels=int(hidden_dim / 2),
									     kernel_size=3, stride=1, padding=1, norm_cfg=batch_norm),  # c0
									dict(type='MaxPool2d', kernel_size=2, stride=2, padding=0),
									dict(type='ConvModule', in_channels=int(hidden_dim / 2), out_channels=hidden_dim,
									     kernel_size=3, stride=1, padding=1, norm_cfg=batch_norm),  # c1
									dict(type='MaxPool2d', kernel_size=2, stride=2, padding=0),  # c2
								],
							),
						),
						collect=dict(type='CollectBlock', from_layer='c2'),
					),
				),
				dict(
					type='SequenceEncoderComponent',
					from_layer='cnn_feat',
					to_layer='src',
					arch=dict(
                        type='TransformerEncoder',
						position_encoder=dict(
							type='Adaptive2DPositionEncoder',
							in_channels=hidden_dim,
							max_h=100,
							max_w=100,
							dropout=dropout,
						),
						encoder_layer=dict(
							type='TransformerEncoderLayer2D',
							attention=dict(
								type='MultiHeadAttention',
								in_channels=hidden_dim,
								k_channels=hidden_dim // n_head,
								v_channels=hidden_dim // n_head,
								n_head=n_head,
								dropout=dropout,
							),
							attention_norm=layer_norm,
							feedforward=dict(
								type='Feedforward',
								layers=[
									dict(type='ConvModule', in_channels=hidden_dim, out_channels=hidden_dim * 4,
									     kernel_size=3, padding=1,
									     bias=True, norm_cfg=None, activation='relu', dropout=dropout),
									dict(type='ConvModule', in_channels=hidden_dim * 4, out_channels=hidden_dim,
									     kernel_size=3, padding=1,
									     bias=True, norm_cfg=None, activation=None, dropout=dropout),
								],
							),
							feedforward_norm=layer_norm,
						),
						num_layers=n_e,
					),
				),
			],
		),
		head=dict(
			type='TransformerHead',
			src_from='src',
			num_steps=num_steps,
			pad_id=num_class,
			decoder=dict(
				type='TransformerDecoder',
				position_encoder=dict(
					type='PositionEncoder1D',
					in_channels=hidden_dim,
					max_len=100,
					dropout=dropout,
				),
				decoder_layer=dict(
					type='TransformerDecoderLayer1D',
					self_attention=dict(
						type='MultiHeadAttention',
						in_channels=hidden_dim,
						k_channels=hidden_dim // n_head,
						v_channels=hidden_dim // n_head,
						n_head=n_head,
						dropout=dropout,
					),
					self_attention_norm=layer_norm,
					attention=dict(
						type='MultiHeadAttention',
						in_channels=hidden_dim,
						k_channels=hidden_dim // n_head,
						v_channels=hidden_dim // n_head,
						n_head=n_head,
						dropout=dropout,
					),
					attention_norm=layer_norm,
					feedforward=dict(
						type='Feedforward',
						layers=[
							dict(type='FCModule', in_channels=hidden_dim, out_channels=hidden_dim * 4, bias=True,
							     activation='relu', dropout=dropout),
							dict(type='FCModule', in_channels=hidden_dim * 4, out_channels=hidden_dim, bias=True,
							     activation=None, dropout=dropout),
						],
					),
					feedforward_norm=layer_norm,
				),
				num_layers=n_d,
			),
			generator=dict(
				type='Linear',
				in_features=hidden_dim,
				out_features=num_class,
			),
			embedding=dict(
				type='Embedding',
				num_embeddings=num_class + 1,
				embedding_dim=hidden_dim,
				padding_idx=num_class,
			),
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
	filter_invalid_indic_labels=True,
	CH=CH,
	V=V,
	v=v,
	m=m,
	symbols=symbols,
)
test_dataset_params = dict(
	batch_max_length=batch_max_length,
	data_filter=True,
	character=test_character,
	filter_invalid_indic_labels=True,
	CH=CH,
	V=V,
	v=v,
	m=m,
	symbols=symbols,
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
## ST dataset
train_root_st = data_root + 'training/ST/'

train_dataset_mj = [dict(type='LmdbDataset', root=train_root_mj + folder_name)
                    for folder_name in mj_folder_names]
train_dataset_st = [dict(type='LmdbDataset', root=train_root_st)]

# valid
valid_root = data_root + 'validation/'
valid_dataset = [dict(type='LmdbDataset', root=valid_root+folder_name, **test_dataset_params)for folder_name in validation_folder_names]

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
			dataset=dict(
				type='ConcatDatasets',
				datasets=valid_dataset,
			),
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
	max_iterations_val = 200,
	snapshot_interval=5000,
	save_best=True,
	resume=None,
)
