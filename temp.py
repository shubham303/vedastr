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