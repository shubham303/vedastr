# work directory
root_workdir = 'workdir'
# sample_per_gpu
samples_per_gpu = 192
###############################################################################
# 1. inference
size = (32, 256)
mean, std = 0.5, 0.5

sensitive = False
character = 'abcdefghijklmnopqrstuvwxyz0123456789'
batch_max_length = 25

fiducial_num = 20
hidden_dim = 256
norm_cfg = dict(type='BN')
num_class = len(character) + 2 # Attention based need two more characters: '[G0]' and '[S]'
num_steps = batch_max_length + 1

dropout = 0.1
n_e = 9
n_d = 3
n_head = 8
batch_norm = dict(type='BN')
layer_norm = dict(type='LayerNorm', normalized_shape=hidden_dim)

test_sensitive = False
test_character = '0123456789abcdefghijklmnopqrstuvwxyz'
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
    ),
    model=dict(
        type='GModel',
        need_text=False,
        body=dict(
            type='GBody',
            pipelines=[
                dict(
                    type="SequenceEncoderComponent",
                    from_layer="input",
                    to_layer="src",
                    arch=dict(
                        type="TransformerEncoder",
                        embedding=dict(
                            type="PatchEmbedding",
                            in_channels=1,
                            patch_height=32,
                            patch_width=8,
                            emb_size=hidden_dim,
                        ),
                        position_encoder=dict(
                            type='PositionEncoder1D',
                            in_channels=hidden_dim,
                            max_len=100,
                            dropout=dropout,
                        ),
                        encoder_layer=dict(
                            type='TransformerEncoderLayer1D',
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
                                    dict(type='FCModule', in_channels=hidden_dim, out_channels=hidden_dim * 4,
                                         bias=True,
                                         activation='relu', dropout=dropout),
                                    dict(type='FCModule', in_channels=hidden_dim * 4, out_channels=hidden_dim,
                                         bias=True,
                                         activation=None, dropout=dropout),
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
            type='VisionTransformerFChead',
            in_channels = hidden_dim,
            num_class=num_class,
            from_layer="src",
            batch_max_length = batch_max_length
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
    character=character,
)

#data_root = './data/data_lmdb_release/'
data_root='/home/shubham/Documents/MTP/datasets/lmdb_datasets/'
###############################################################################
# 3. test
test_root = data_root + 'evaluation/'
#test_folder_names = ['CUTE80', 'IC03_867', 'IC13_1015', 'IC15_2077',
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
        transform=inference['transform'],
    ),
    postprocess_cfg=dict(
        sensitive=sensitive,
        character=character,
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
valid_dataset = dict(type='LmdbDataset', root=valid_root, **dataset_params)

# train transforms
train_transforms = [
    dict(type='Sensitive', sensitive=sensitive),
    dict(type='Filter', need_character=character),
    dict(type='ToGray'),
    dict(type='Resize', size=size),
    dict(type='Normalize', mean=mean, std=std),
    dict(type='ToTensor'),
]

max_iterations = 300000
milestones = [150000, 250000]

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
            transform=inference['transform'],
        ),
    ),
    optimizer=dict(type='Adadelta', lr=1.0, rho=0.95, eps=1e-8),
    criterion=dict(type='CrossEntropyLoss'),
    lr_scheduler=dict(type='StepLR',
                      milestones=milestones,
                      ),
    max_iterations=max_iterations,
    log_interval=10,
    trainval_ratio=2000,
    snapshot_interval=20000,
    save_best=True,
    resume=None,
)
