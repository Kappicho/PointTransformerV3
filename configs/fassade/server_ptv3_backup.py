_base_ = ["../../configs/_base_/default_runtime.py"]

load_from = "exp/scannet/pretrained/scannet-ptv3.pth"

# --- CONFIG FOR GUARANTEED RUN ---
batch_size = 1
num_worker = 1       # Zurück auf 1 (Standard)
mix_prob = 0.8
empty_cache = True
enable_amp = True
enable_wandb = False 

# --- KEINE VALIDIERUNG = KEIN ABSTURZ ---
evaluate = False     
epoch = 400
eval_epoch = 400     

model = dict(
    type="DefaultSegmentorV2",
    num_classes=9,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1",
        in_channels=7,
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.5, ignore_index=-1),
    ],
)

optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.05)
scheduler = dict(type="OneCycleLR", max_lr=0.002, pct_start=0.04, anneal_strategy="cos", div_factor=10.0, final_div_factor=10000.0)

dataset_type = "DefaultDataset"
data_root = "data/fassade"

data = dict(
    num_classes=9,
    ignore_index=-1,
    names=["Sonstiges", "Putz", "Beton", "Backstein", "Dachziegel", "SteinNatur", "SteinFliesen", "Kunststoff", "Metall"],
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="GridSample", grid_size=0.03, hash_type="fnv", mode="train", return_grid_coord=True),
            # TRAINING: 300k Points
            dict(type="SphereCrop", point_max=300000, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "grid_coord", "segment"), feat_keys=("color", "normal", "strength")),
        ],
        test_mode=False,
        loop=30,  # Jetzt sollte loop=30 greifen, weil die Dateien gefunden werden!
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="GridSample", grid_size=0.03, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="SphereCrop", point_max=800000, mode="center"), 
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "grid_coord", "segment"), feat_keys=("color", "normal", "strength")),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="GridSample", grid_size=0.03, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
        ],
        test_mode=True,
    ),
)

hooks = [
    dict(type='CheckpointLoader'),
    dict(type='ModelHook'),
    dict(type='IterationTimer', warmup_iter=2),
    dict(type='InformationWriter'),
    dict(type='CheckpointSaver', save_freq=None)
]
