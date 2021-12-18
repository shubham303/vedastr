# language specific changes:
character = 'ऀँंःऄअआइईउऊऋऌऍऎएऐऑऒओऔकखगघङचछजझञटठडढणतथदधनऩपफबभमयरऱलळऴवशषसहऺऻ़ऽािीुूृॄॅॆेैॉॊोौ्ॎॏॐ॒॑॓॔ॕॖॗक़ख़ग़ज़ड़ढ़फ़य़ॠॡॢॣ।॥०१२३४५६७८९ॲ%/?:,.-'
test_sensitive = False
test_character = 'ऀँंःऄअआइईउऊऋऌऍऎएऐऑऒओऔकखगघङचछजझञटठडढणतथदधनऩपफबभमयरऱलळऴवशषसहऺऻ़ािीुूृॄॅॆेैॉॊोौ्ॎॏॐ॒॑॓॔ॕॖॗक़ख़ग़ज़ड़ढ़फ़य़ॠॡॢॣ०१२३४५६७८९ॲ'
batch_max_length = 25
test_folder_names = ['IIIT']  ###
data_root = '/usr/datasets/synthetic_text_dataset/lmdb_dataset_Hindi/'
# data_root = '/home/ocr/datasets/recognition/hindi/'
validation_folder_names = ['MJ_valid', "ST_valid"]
mj_folder_names = ['MJ_test', 'MJ_train']

## MJ dataset
train_root_mj = data_root + 'training/MJ/'

## ST dataset
train_root_st = data_root + 'training/ST/'

train_dataset_mj = [dict(type='LmdbDataset', root=train_root_mj + folder_name)
                    for folder_name in mj_folder_names]
train_dataset_st = [dict(type='LmdbDataset', root=train_root_st)]

# valid

valid_root = data_root + 'validation/'
valid_dataset = [dict(type='LmdbDataset', root=valid_root + folder_name, **test_dataset_params) for folder_name in
                 validation_folder_names]

# CH V , v and m are used to filter valid words in language
m = "ऀ  ँ ं ः  ॕ "
V = "ऄ ई ऊ ऍ  ऎ ऐ ऑ ऒ ओ औ"
CH = "अ आ उ ए इ ऌ क  ख  ग ऋ  घ  ङ  च  छ  ज  झ  ञ  ट  ठ  ड  ढ  ण  त  थ  द  ध  न  ऩ  प  फ  ब  भ  म  य  र  ऱ  ल  ळ  ऴ  व  " \
     "श  ष  " \
     "स  ह ॐ क़  ख़  ग़  ज़  ड़  ढ़  फ़  य़  ॠ  ॡ"
v = "ा  ि  ी  ु  ू  ृ  ॄ  ॉ  ॊ  ो  ौ  ॎ  ॏ ॑  ॒  ॓ ़ ॔  ॅ े ै ॆ ्  ॖ   ॗ ॢ  ॣ"
symbols = "।  ॥  ०  १  २  ३  ४  ५  ६  ७  ८  ९ %  /  ?  :  ,  .  -"

# work directory
root_workdir = 'workdir'
# sample_per_gpu
samples_per_gpu = 64
###############################################################################
# 1. inference
size = (32, 100)
mean, std = 0.5, 0.5

sensitive = True
fiducial_num = 20
dropout = 0.1
n_e = 9
n_d = 3
n_head = 8
norm_cfg = dict(type='BN')
num_characters = len(character) + 2  # extra go and end character.
num_class = len(character) + 1  # [GO] character is not in prediction list.

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
		vis_module=dict(
			type="VisualModule",
			tps=dict(
				type='RectificatorComponent',
				from_layer='input',
				to_layer='rect',
				arch=dict(
					type='TPS_STN',
					F=fiducial_num,
					input_size=size,
					output_size=size,
					stn=dict(
						feature_extractor=dict(
							encoder=dict(
								backbone=dict(
									type='GBackbone',
									layers=[
										dict(type='ConvModule', in_channels=1, out_channels=64,
										     kernel_size=3, stride=1, padding=1, norm_cfg=norm_cfg),
										dict(type='MaxPool2d', kernel_size=2, stride=2),
										dict(type='ConvModule', in_channels=64, out_channels=128,
										     kernel_size=3, stride=1, padding=1, norm_cfg=norm_cfg),
										dict(type='MaxPool2d', kernel_size=2, stride=2),
										dict(type='ConvModule', in_channels=128, out_channels=256,
										     kernel_size=3, stride=1, padding=1, norm_cfg=norm_cfg),
										dict(type='MaxPool2d', kernel_size=2, stride=2),
										dict(type='ConvModule', in_channels=256, out_channels=512,
										     kernel_size=3, stride=1, padding=1, norm_cfg=norm_cfg),
									],
								),
							),
							collect=dict(type='CollectBlock', from_layer='c3')
						),
						pool=dict(type='AdaptiveAvgPool2d', output_size=1),
						head=[
							dict(type='FCModule', in_channels=512, out_channels=256),
							dict(type='FCModule', in_channels=256, out_channels=fiducial_num * 2, activation=None)
						],
					),
				),
			),
			d_input=1,
			layers=[3, 4, 6, 6, 3],
			n_layer=3,
			d_model=512,
			d_inner=1024,
			n_head=8,
			d_k=64,
			d_v=64,
			dropout=0
		),
		pos_module=dict(
			type="PositionalEmbedding",
			d_onehot=512,
			d_hid=512,
			n_position=200,
			max_seq_len=batch_max_length
		),
		sem_module=dict(
			type="SemanticEmbedding",
			d_model=512,
			rnn_layers=2,
			rnn_dropout=0,
			d_k=64,
			attn_dropout=0,
			max_seq_len=batch_max_length,
			padding_idx=num_class,
			num_classes=num_characters
		),
		mdcdp_layers=[dict(
			type="MDCDP",
			n_layer_sae=1,
			d_model_sae=512,
			d_inner_sae=1024,
			n_head_sae=8,
			d_k_sae=64,
			d_v_sae=64,
			n_layer=3,
			d_model=512,
			d_inner=512,
			n_head=8,
			d_k=64,
			d_v=64,
			dropout=0
		) for i in range(0, 3)],
		
		need_text=True,
		max_seq_len=batch_max_length + 1,
		d_model=512,
		num_class=num_class
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


train_transforms = [
	dict(type='Sensitive', sensitive=sensitive),
	dict(type='Filter', need_character=character),
	dict(type='ToGray'),
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
	optimizer=dict(type='Adam', lr=0.001),
	criterion=dict(type='CrossEntropyLoss'),
	lr_scheduler=dict(type='CosineLR',
	                  iter_based=True,
	                  warmup_epochs=0.1,
	                  ),
	max_epochs=max_epochs,
	log_interval=10,
	trainval_ratio=4000,
	max_iterations_val=200,  # 10 percent of train_val ratio.
	snapshot_interval=5000,
	save_best=True,
	resume=None
	# resume=None
)
