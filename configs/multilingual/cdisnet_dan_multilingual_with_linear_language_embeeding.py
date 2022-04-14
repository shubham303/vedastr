# language specific changes:
import os.path

from vedastr.attention_masks.masks import generate_square_subsequent_mask, src_mask_attend_only_neighbour_tokens, \
	diagonal_mask

character = "ஂஃஅஆஇஈஉஊஎஏஐஒஓஔக஗ஙசஜஞடணதநனப஬மயரறலளழவஶஷஸஹ஻஼஽ாிீுூெேைொோௌ்௏ௐௗ௘௛௞௦௧௨௩௪௫௬௭௮௯௰௱௲௳௴௵௶௷௸௹௺ഀഁംഃഄഅആഇഈഉഊഋഌഎഏഐഒഓഔകഖഗഘങചഛജഝഞടഠഡഢണതഥദധനഩപഫബഭമയരറലളഴവശഷസഹഺ഻഼ഽാിീുൂൃൄെേൈൊോൌ്ൎ൏ൔൕൖൗ൘൙൚൛൜൝൞ൟൠൡൢൣ൦൧൨൩൪൫൬൭൮൯൰൱൲൳൴൵൶൷൸൹ൺൻർൽൾൿঀঁংঃঅআইঈউঊঋঌএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরল঳঴঵শষসহ়ঽািীুূৃৄেৈোৌ্ৎৗড়ঢ়য়ৠৡৢৣ০১২৩৪৫৬৭৮৯ৰৱ৲৳৴৵৶৷৸৹৺৻ৼ৽৾ఁంఃఄఅఆఇఈఉఊఋఌఎఏఐఒఓఔకఖగఘఙచఛజఝఞటఠడఢణతథదధనపఫబభమయరఱలళఴవశషసహఽాిీుూృౄెేైొోౌ్ౕౖౘౙౚౠౡౢౣ౦౧౨౩౪౫౬౭౮౯౱౷౸౹౺౻౼౽౾౿ऀँंःऄअआइईउऊऋऌऍऎएऐऑऒओऔकखगघङचछजझञटठडढणतथदधनऩपफबभमयरऱलळऴवशषसहऺऻ़ऽािीुूृॄॅॆेैॉॊोौ्ॎॏॐ॒॑॓॔ॕॖॗक़ख़ग़ज़ड़ढ़फ़य़ॠॡॢॣ०१२३४५६७८९ॲಀಁಂಃ಄ಅಆಇಈಉಊಋಌಎಏಐಒಓಔಕಖಗಘಙಚಛಜಝಞಟಠಡಢಣತಥದಧನಪಫಬಭಮಯರಱಲಳವಶಷಸಹ಼ಽಾಿೀುೂೃೄ೅ೆೇೈ೉ೊೋೌ್ೕೖೞೠೡೢೣ೦೧೨೩೪೫೬೭೮೯ଁଂଃଅଆଇଈଉଊଋଌଏଐଓଔକଖଗଘଙଚଛଜଝଞଟଠଡଢଣତଥଦଧନପଫବଭମଯର଱ଲଳଵଶଷସହ଼ଽାିୀୁୂୃୄେୈୋୌ୍୐ୖୗଡ଼ଢ଼ୟୠୡୢୣ୤୦୧୨୩୪୫୬୭୮୯୰ୱ୲୳୴୵୶୷ਁਂਃਅਆਇਈਉਊਏਐਓਔਕਖਗਘਙਚਛਜਝਞਟਠਡਢਣਤਥਦਧਨਪਫਬਭਮਯਰਲਲ਼ਵਸ਼ਸਹ਼ਾਿੀੁੂੇੈੋੌ੍ੑਖ਼ਗ਼ਜ਼ੜਫ਼੦੧੨੩੪੫੬੭੮੯ੰੱੲੳੴੵ੶ઁંઃઅઆઇઈઉઊઋઌઍએઐઑઓઔકખગઘઙચછજઝઞટઠડઢણતથદધનપફબભમયરલળવશષસહ઺઼ઽાિીુૂૃૄૅેૈૉોૌ્ૐૠૡૢૣ૤૥૦૧૨૩૪૫૬૭૮૯"
test_sensitive = False
test_character = "ஂஃஅஆஇஈஉஊஎஏஐஒஓஔக஗ஙசஜஞடணதநனப஬மயரறலளழவஶஷஸஹ஻஼஽ாிீுூெேைொோௌ்௏ௐௗ௘௛௞௦௧௨௩௪௫௬௭௮௯௰௱௲௳௴௵௶௷௸௹௺ഀഁംഃഄഅആഇഈഉഊഋഌഎഏഐഒഓഔകഖഗഘങചഛജഝഞടഠഡഢണതഥദധനഩപഫബഭമയരറലളഴവശഷസഹഺ഻഼ഽാിീുൂൃൄെേൈൊോൌ്ൎ൏ൔൕൖൗ൘൙൚൛൜൝൞ൟൠൡൢൣ൦൧൨൩൪൫൬൭൮൯൰൱൲൳൴൵൶൷൸൹ൺൻർൽൾൿঀঁংঃঅআইঈউঊঋঌএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরল঳঴঵শষসহ়ঽািীুূৃৄেৈোৌ্ৎৗড়ঢ়য়ৠৡৢৣ০১২৩৪৫৬৭৮৯ৰৱ৲৳৴৵৶৷৸৹৺৻ৼ৽৾ఁంఃఄఅఆఇఈఉఊఋఌఎఏఐఒఓఔకఖగఘఙచఛజఝఞటఠడఢణతథదధనపఫబభమయరఱలళఴవశషసహఽాిీుూృౄెేైొోౌ్ౕౖౘౙౚౠౡౢౣ౦౧౨౩౪౫౬౭౮౯౱౷౸౹౺౻౼౽౾౿ऀँंःऄअआइईउऊऋऌऍऎएऐऑऒओऔकखगघङचछजझञटठडढणतथदधनऩपफबभमयरऱलळऴवशषसहऺऻ़ऽािीुूृॄॅॆेैॉॊोौ्ॎॏॐ॒॑॓॔ॕॖॗक़ख़ग़ज़ड़ढ़फ़य़ॠॡॢॣ०१२३४५६७८९ॲಀಁಂಃ಄ಅಆಇಈಉಊಋಌಎಏಐಒಓಔಕಖಗಘಙಚಛಜಝಞಟಠಡಢಣತಥದಧನಪಫಬಭಮಯರಱಲಳವಶಷಸಹ಼ಽಾಿೀುೂೃೄ೅ೆೇೈ೉ೊೋೌ್ೕೖೞೠೡೢೣ೦೧೨೩೪೫೬೭೮೯ଁଂଃଅଆଇଈଉଊଋଌଏଐଓଔକଖଗଘଙଚଛଜଝଞଟଠଡଢଣତଥଦଧନପଫବଭମଯର଱ଲଳଵଶଷସହ଼ଽାିୀୁୂୃୄେୈୋୌ୍୐ୖୗଡ଼ଢ଼ୟୠୡୢୣ୤୦୧୨୩୪୫୬୭୮୯୰ୱ୲୳୴୵୶୷ਁਂਃਅਆਇਈਉਊਏਐਓਔਕਖਗਘਙਚਛਜਝਞਟਠਡਢਣਤਥਦਧਨਪਫਬਭਮਯਰਲਲ਼ਵਸ਼ਸਹ਼ਾਿੀੁੂੇੈੋੌ੍ੑਖ਼ਗ਼ਜ਼ੜਫ਼੦੧੨੩੪੫੬੭੮੯ੰੱੲੳੴੵ੶ઁંઃઅઆઇઈઉઊઋઌઍએઐઑઓઔકખગઘઙચછજઝઞટઠડઢણતથદધનપફબભમયરલળવશષસહ઺઼ઽાિીુૂૃૄૅેૈૉોૌ્ૐૠૡૢૣ૤૥૦૧૨૩૪૫૬૭૮૯"

batch_max_length = 25
test_folder_names = ["kaggle_train", "kaggle_val", "icdar"]  ###

languages = ["HI", "ML", "KN", "TA", "TE", "OR", "GUR", "GU", "BN"]  # note these language codes should match with
# language
# code returned by abfn module.

data_roots = ['/media/shubham/One Touch/Indic_OCR/recognition_dataset/tamil',
              '/media/shubham/One Touch/Indic_OCR/recognition_dataset/bengali',
              '/media/shubham/One Touch/Indic_OCR/recognition_dataset/telugu',
              '/media/shubham/One Touch/Indic_OCR/recognition_dataset/hindi',
              '/media/shubham/One Touch/Indic_OCR/recognition_dataset/malayalam',
              '/media/shubham/One Touch/Indic_OCR/recognition_dataset/gujarati',
              '/media/shubham/One Touch/Indic_OCR/recognition_dataset/gurumukhi',
              '/media/shubham/One Touch/Indic_OCR/recognition_dataset/kannada',
              '/media/shubham/One Touch/Indic_OCR/recognition_dataset/oriya']

# data_root = '/home/ocr/datasets/recognition/hindi/'
# data_root= '/nlsasfs/home/ai4bharat/shubhamr/shubham/recognition-dataset/hindi/'

validation_folder_names = ["MJ_valid", "ST_valid"]
#validation_folder_names = ["IIIT"]
# validation_folder_names = ["kaggle_train", "kaggle_val", "1", "2", "3", "4", "5", "6", "7"]
mj_folder_names = ['MJ_train']

real_world_train_folders = ["kaggle_train", "kaggle_val", "1", "2", "3", "4", "5", "6", "7", "icdar"]

##############################################################################################
# dataset related configuration.
fine_tune = False  # set to true to finetune model on real dataset.
train_datasets = []
valid_datasets = []
test_datasets = []
for root in data_roots:
	
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
	
	try:
		if not fine_tune:
			st = os.path.join(root, "training/ST")
			mj = os.path.join(root, "training/MJ/")
			
			train_dataset_mj = [mj + folder_name for folder_name in mj_folder_names]
			train_dataset_mj = list(filter(lambda x: os.path.exists(x), train_dataset_mj))
			train_dataset_mj = [dict(type='LmdbDataset', root=folder_name)
			                    for folder_name in train_dataset_mj]
			
			train_dataset_st = []
			if os.path.exists(st):
				train_dataset_st = [dict(type='LmdbDataset', root=st)]
			
			train_datasets.append(train_dataset_mj)
			train_datasets.append(train_dataset_st)
		
		else:
			train_root_real = os.path.join(root, "evaluation/")
			
			train_dataset_real = [train_root_real + folder_name for folder_name in real_world_train_folders]
			train_dataset_real = list(filter(lambda x: os.path.exists(x), train_dataset_real))
			train_dataset_real = [dict(type='LmdbDataset', root=folder_name)
			                      for folder_name in train_dataset_real]
			
			if len(train_dataset_real) > 0:
				train_datasets.append(train_dataset_real)
		
		valid_root = os.path.join(root, 'validation/')
		
		valid_dataset = [valid_root + folder_name for folder_name in validation_folder_names]
		valid_dataset = list(filter(lambda x: os.path.exists(x), valid_dataset))
		valid_dataset = [dict(type='LmdbDataset', root=folder_name, **test_dataset_params) for
		                 folder_name in valid_dataset]
		
		if len(valid_dataset) > 0:
			valid_datasets.append(valid_dataset)
		
		test_root = os.path.join(root, "evaluation/")
		
		test_dataset = [test_root + folder_name for folder_name in test_folder_names]
		test_dataset = list(filter(lambda x: os.path.exists(x), test_dataset))
		test_dataset = [dict(type='LmdbDataset', root=f_name, **test_dataset_params) for f_name in
		                test_dataset]
		
		test_datasets.extend(test_dataset)
	
	
	except Exception:
		""" Note : ("exception occurred during dataset creation. For multilingual model this exception occurs because
		# some ""langauge dataset may not have dataset with specific name. if that is the case no need to worry.")
		"""
		print(Exception)
		continue

##############################################################################################


# work directory
root_workdir = '/media/shubham/One Touch/Indic_OCR/models/'
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
layer_norm_cfg = dict(type="LN")
num_characters = len(character) + 2  # extra go and end character.
num_class = len(character) + 1  # [GO] character is not in prediction list.
hidden_dim = 512
hidden_dim_cbi = hidden_dim
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
		language_list=languages  # language id is returned by abfn module
	),
	model=dict(
		type='Cdisnet',
		vis_module=dict(
			type="GBody",
			pipelines=[
				dict(
					type='FeatureExtractorComponent',
					from_layer='input',
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
						in_channels=hidden_dim,
						max_len=200,
						dropout=dropout
					)
				)
			
			],
			collect=dict(type='CollectBlock', from_layer='vis_positional_encoding'),
		),
		pos_module=dict(
			type="GBody",
			pipelines=[
				dict(
					type="PositionalEncodingComponent",
					from_layer="input",
					to_layer="positional_embedding",
					
					arch=dict(
						type="PositionEncoder1D",
						in_channels=hidden_dim,
						max_len=200,
						dropout=dropout
					),
				
				),
				dict(
					type="PlugComponent",
					from_layer="positional_embedding",
					to_layer="linear_layer",
					arch=dict(type='FCModules', in_channels=hidden_dim, out_channels=hidden_dim, activation=
					"relu", num_fcs=2, norm=layer_norm_cfg)),
			
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
						in_channels=hidden_dim,
						max_len=200,
						dropout=dropout
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
					num_layers=3,
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
					num_layers=5,
					position_encoder=None,
					embedding=None
				),
				d_model=hidden_dim,
				pos_mask=generate_square_subsequent_mask,
				vis_mask=diagonal_mask,
				sem_mask=generate_square_subsequent_mask,
				vis_mask_range=2,
				sem_mask_range=7,
				activation=dict(
					type="Sigmoid"
				)
			)
			for i in range(0, 5)],
		language_embedding=dict(
			type="GBody",
			pipelines=[
				dict(
					type='EmbeddingComponent',
					from_layer='input',
					to_layer='language_embedding',
					arch=dict(
						type='Embedding',
						num_embeddings=len(languages) + 1,  # num_embedding is 1 extra than total
						# number of
						# languages.
						embedding_dim=hidden_dim,  #
					)
				),
				dict(
					type="PlugComponent",
					from_layer="language_embedding",
					to_layer="linear_layer",
					arch=dict(type='FCModule', in_channels=hidden_dim, out_channels=hidden_dim, dropout=dropout)),
			],
			collect=dict(type='CollectBlock', from_layer='linear_layer')
		),
		linear_layer =dict(
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
		need_text=True,
		need_lang=True,
		max_seq_len=batch_max_length + 1,
		d_model=hidden_dim,
		num_class=num_class,
		share_weight=False
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


# data_root = './data/data_lmdb_release/'
###############################################################################


test = dict(
	data=dict(
		dataloader=dict(
			type='DataLoader',
			samples_per_gpu=samples_per_gpu,
			workers_per_gpu=4,
			shuffle=False,
		),
		sampler=dict(type='DefaultSampler', shuffle=False),
		dataset=test_datasets,
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


# valid


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
				datasets=[dict(type="ConcatDatasets", datasets=d) for d in train_datasets],
				batch_ratio=[1 / len(train_datasets)] * len(train_datasets),  # this batch ratio reads
				# data
				# from every dataset with equal size.
				**dataset_params,
			),
			transform=train_transforms,
		),
		val=dict(
			dataloader=dict(
				type='DataLoader',
				samples_per_gpu=samples_per_gpu,
				workers_per_gpu=4,
				shuffle=True,
			),
			dataset=dict(
				type='ConcatDatasets',
				datasets=[dict(type="ConcatDatasets", datasets=d) for d in valid_datasets],
			),
			transform=test['data']['transform'],
		)
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