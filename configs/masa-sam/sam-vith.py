prompt_embed_dim=256
model = dict(
    type='SamMasa',
    backbone=dict(
        type='ImageEncoderViT',
        depth=32,
        embed_dim=1280,
        img_size=1024,
        mlp_ratio=4,
        num_heads=16,
        patch_size=16,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=[7, 15, 23, 31],
        window_size=14,
        out_chans=prompt_embed_dim,
        out_indices=[7, 15, 23, 31]),
    mask_decoder=dict(
        type='MaskDecoder',
        num_multimask_outputs=3,
        transformer_dim=prompt_embed_dim,
        iou_head_depth=3,
        iou_head_hidden_dim=256),
    prompt_encoder=dict(
        type='PromptEncoder',
        embed_dim=prompt_embed_dim,
        image_embedding_size=(64, 64),
        input_image_size=(1024, 1024),
        mask_in_chans=16),
)