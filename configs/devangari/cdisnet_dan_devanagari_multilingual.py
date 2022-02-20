# language specific changes:
from vedastr.attention_masks.masks import generate_square_subsequent_mask, src_mask_attend_only_neighbour_tokens

character = 'ऀँंःऄअआइईउऊऋऌऍऎएऐऑऒओऔकखगघङचछजझञटठडढणतथदधनऩपफबभमयरऱलळऴवशषसहऺऻ़ऽािीुूृॄॅॆेैॉॊोौ्ॎॏॐ॒॑॓॔ॕॖॗक़ख़ग़ज़ड़ढ़फ़य़ॠॡॢॣ०१२३४५६७८९ॲഀഁംഃഄഅആഇഈഉഊഋഌഎഏഐഒഓഔകഖഗഘങചഛജഝഞടഠഡഢണതഥദധനഩപഫബഭമയരറലളഴവശഷസഹഺ഻഼ഽാിീുൂൃൄെേൈൊോൌ്ൎ൏ൔൕൖൗ൘൙൚൛൜൝൞ൟൠൡൢൣ൦൧൨൩൪൫൬൭൮൯൰൱൲൳൴൵൶൷൸൹ൺൻർൽൾൿ'
test_sensitive = False
test_character = 'ऀँंऄअआइईउऊऋऌऍऎएऐऑऒओऔकखगघङचछजझञटठडढणतथदधनऩपफबभमयरऱलळऴवशषसहऺऻ़ािीुूृॄॅॆेैॉॊोौ्ॎॏॐ॒॑॓॔ॕॖॗक़ख़ग़ज़ड़ढ़फ़य़ॠॡॢॣ०१२३४५६७८९ॲഀഁംഃഄഅആഇഈഉഊഋഌഎഏഐഒഓഔകഖഗഘങചഛജഝഞടഠഡഢണതഥദധനഩപഫബഭമയരറലളഴവശഷസഹഺ഻഼ഽാിീുൂൃൄെേൈൊോൌ്ൎ൏ൔൕൖൗ൘൙൚൛൜൝൞ൟൠൡൢൣ൦൧൨൩൪൫൬൭൮൯൰൱൲൳൴൵൶൷൸൹ൺൻർൽൾൿ'
batch_max_length = 25
test_folder_names = ["2", "3", "4", "5", "6", "7"]  ###

data_root = '/usr/datasets/synthetic_text_dataset/lmdb_dataset/hindi/'
data_root_1 = '/usr/datasets/synthetic_text_dataset/lmdb_dataset/malayalam/'

# data_root = '/home/ocr/datasets/recognition/hindi/'
# data_root= '/nlsasfs/home/ai4bharat/shubhamr/shubham/recognition-dataset/hindi/'
validation_folder_names = ["MJ_valid", "ST_valid"]
# validation_folder_names= [ "1","2","3", "4", "5", "6", "7" ]
mj_folder_names = ['MJ_test', 'MJ_train']

real_world_train_folders = ["IIIT", "kaggle_train", "kaggle_val", "icdar_hindi", "1"]

# CH V , v and m are used to filter valid words in language
# work directory
root_workdir = 'workdir'
# sample_per_gpu
samples_per_gpu = 64
###############################################################################
# 1. inference
size = (32, 128)
mean, std = 0.5, 0.5

sensitive = True
fiducial_num = 20
dropout = 0.1
norm_cfg = dict(type='BN')
num_characters = len(character) + 2  # extra go and end character.
num_class = len(character) + 1  # [GO] character is not in prediction list.
hidden_dim = 512
hidden_dim_cbi = 512
n_head = 8
layer_norm = dict(type='LayerNorm', normalized_shape=hidden_dim)
layer_norm_cbi = dict(type='LayerNorm', normalized_shape=hidden_dim)

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
		language_dict={"HI":0, "ML": 1, "BN": 2 , "TA":3 }      # language dict is used to convert lanuage id returned
		# by abfn to numeric value.
	),
	model=dict(
		type='Cdisnet',
		vis_module=dict(
			type="GBody",
			pipelines=[
				dict(
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
				dict(
					type='FeatureExtractorComponent',
					from_layer='rect',
					to_layer='fpn_feat',
					arch=dict(
						encoder=dict(
							backbone=dict(
								type="FPN",
								strides=[(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)],
								compress_layer=False,
								input_shape=[1, size[0], size[1]],
								maxT=batch_max_length + 1,
								depth=8,
								num_channels=64
							),  # c1
						),
						collect=dict(type='CollectBlock', from_layer='c1'),
					)
				),
				dict(
					type="PositionalEncodingComponent",
					from_layer='fpn_feat',
					to_layer="vis_positional_encoding",
					arch=dict(
						type="PositionEncoder1D",
						in_channels=512,
						max_len=200,
						dropout=0.1
					)
				)
			
			],
			collect=dict(type='CollectBlock', from_layer='vis_positional_encoding'),
		),
		pos_module=dict(
			type="GBody",
			pipelines=[
				dict(
					type="EmbeddingComponent",
					from_layer="input",
					to_layer="positional_embedding",
					arch=dict(
						type="PositionalEmbedding",
						d_onehot=512,
						positional_encoding=dict(
							type="PositionEncoder1D",
							in_channels=512,
							max_len=200,
							dropout=0.1
						)
					),
				)
			],
			collect=dict(type='CollectBlock', from_layer='positional_embedding'),
		),
		sem_module=dict(
			type="GBody",
			pipelines=[
				dict(
					type='EmbeddingComponent',
					from_layer='input',
					to_layer='semantic_embedding',
					arch=dict(
						type='Embedding',
						num_embeddings=num_characters,
						embedding_dim=hidden_dim,
						padding_idx=num_class,
					)
				),
				dict(
					type="PositionalEncodingComponent",
					from_layer='semantic_embedding',
					to_layer="semantic_pos_encoding",
					arch=dict(
						type="PositionEncoder1D",
						in_channels=512,
						max_len=200,
						dropout=0.1
					)
				)
			],
			collect=dict(type='CollectBlock', from_layer='semantic_pos_encoding'),
		
		),
		mdcdp_layers=[
			dict(
				type="MDCDP",
				sae_config=dict(
					type="TransformerUnit",
					encoder_layer=dict(
						type="TransformerEncoderLayer1D",
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
								dict(type='FCModule', in_channels=hidden_dim, out_channels=hidden_dim * 2, bias=True,
								     activation='relu', dropout=dropout),
								dict(type='FCModule', in_channels=hidden_dim * 2, out_channels=hidden_dim, bias=True,
								     activation=None, dropout=dropout),
							],
						),
						feedforward_norm=layer_norm,
					),
					num_layers=1,
					position_encoder=None,
					embedding=None
				),
				cbi_s_config=dict(
					type="TransformerUnit",
					encoder_layer=dict(
						type="TransformerEncoderLayer1D",
						attention=dict(
							type='MultiHeadAttention',
							in_channels=hidden_dim_cbi,
							k_channels=hidden_dim_cbi // n_head,
							v_channels=hidden_dim_cbi // n_head,
							n_head=n_head,
							dropout=dropout,
						),
						attention_norm=layer_norm_cbi,
						feedforward=dict(
							type='Feedforward',
							layers=[
								dict(type='FCModule', in_channels=hidden_dim_cbi, out_channels=hidden_dim_cbi * 2,
								     bias=True,
								     activation='relu', dropout=dropout),
								dict(type='FCModule', in_channels=hidden_dim_cbi * 2, out_channels=hidden_dim_cbi,
								     bias=True,
								     activation=None, dropout=dropout),
							],
						),
						feedforward_norm=layer_norm_cbi,
					),
					num_layers=3,
					position_encoder=None,
					embedding=None
				),
				cbi_v_config=dict(
					type="TransformerUnit",
					encoder_layer=dict(
						type="TransformerEncoderLayer1D",
						attention=dict(
							type='MultiHeadAttention',
							in_channels=hidden_dim_cbi,
							k_channels=hidden_dim_cbi // n_head,
							v_channels=hidden_dim_cbi // n_head,
							n_head=n_head,
							dropout=dropout,
						),
						attention_norm=layer_norm_cbi,
						feedforward=dict(
							type='Feedforward',
							layers=[
								dict(type='FCModule', in_channels=hidden_dim_cbi, out_channels=hidden_dim_cbi * 2,
								     bias=True,
								     activation='relu', dropout=dropout),
								dict(type='FCModule', in_channels=hidden_dim_cbi * 2, out_channels=hidden_dim_cbi,
								     bias=True,
								     activation=None, dropout=dropout),
							],
						),
						feedforward_norm=layer_norm_cbi,
					),
					num_layers=3,
					position_encoder=None,
					embedding=None
				),
				d_model=hidden_dim,
				pos_mask=generate_square_subsequent_mask,
				vis_mask=src_mask_attend_only_neighbour_tokens,
				sem_mask=generate_square_subsequent_mask,
				activation=dict(
					type="Sigmoid"
				)
			)
			for i in range(0, 1)],
		
		language_embedding=dict(
			type="LanguageEmbedding",
			embedding=dict(
				type='Embedding',
				num_embeddings=2,
				embedding_dim=hidden_dim,
			),
			d_model=hidden_dim
		),
		
		need_text=True,
		max_seq_len=batch_max_length + 1,
		d_model=hidden_dim,
		num_class=num_class,
		share_weight=True
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
	data_filter=True,
	character=test_character,
)

# data_root = './data/data_lmdb_release/'
###############################################################################
# 3. test
test_root = data_root + 'evaluation/'

dict(
	type='ConcatDatasets',
	datasets=[dict(type="ConcatDatasets", datasets=d) for d in valid_datasets]
	         ** test_dataset_params,
),
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

train_root_mj_1 = data_root_1 + 'training/MJ/'

## ST dataset
train_root_st_1 = data_root_1 + 'training/ST/'

train_dataset_mj = [dict(type='LmdbDataset', root=train_root_mj + folder_name)
                    for folder_name in mj_folder_names]
train_dataset_st = [dict(type='LmdbDataset', root=train_root_st)]

train_dataset_mj_1 = [dict(type='LmdbDataset', root=train_root_mj_1 + folder_name)
                      for folder_name in mj_folder_names]
train_dataset_st_1 = [dict(type='LmdbDataset', root=train_root_st_1)]

train_root_real = data_root + "evaluation/"

train_dataset_real = [dict(type='LmdbDataset', root=train_root_real + folder_name)
                      for folder_name in real_world_train_folders]

# valid

valid_root = data_root + 'validation/'
valid_dataset = [dict(type='LmdbDataset', root=valid_root + folder_name, **test_dataset_params) for folder_name in
                 validation_folder_names]

valid_root_1 = data_root + 'validation/'
valid_dataset_1 = [dict(type='LmdbDataset', root=valid_root_1 + folder_name, **test_dataset_params) for folder_name in
                   validation_folder_names]

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
						# datasets=train_dataset_real
					),
					dict(
						type='ConcatDatasets',
						datasets=train_dataset_st,
						# datasets=train_dataset_real
					),
					
					dict(
						type='ConcatDatasets',
						datasets=train_dataset_mj_1,
						# datasets=train_dataset_real
					),
					dict(
						type='ConcatDatasets',
						datasets=train_dataset_st_1,
						# datasets=train_dataset_real
					),
				],
				batch_ratio=[0.25, 0.25, 0.25, 0.25],
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
				datasets=[
					dict(
						type='ConcatDatasets',
						datasets=valid_dataset,
					),
					dict(
						type='ConcatDatasets',
						datasets=valid_dataset_1,
					),
				]),
			
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
	log_interval=50,
	trainval_ratio=5000,
	max_iterations_val=500,  # 10 percent of train_val ratio.
	snapshot_interval=10000,
	save_best=True,
	resume=None
)
