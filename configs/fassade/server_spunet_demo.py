_base_ = ["../../configs/_base_/default_runtime.py"]

load_from = "exp/scannet/pretrained/scannet-spunet.pth"

# --- SETTINGS ---
batch_size = 1
num_worker = 1
mix_prob = 0.8
empty_cache = True
enable_amp = False 
enable_wandb=False

# --- TRAINING ---
evaluate = True
epoch = 10
eval_epoch = 10

# Gewichtung
class_weights = [0.1, 1.0, 3.0, 2.0, 3.0, 2.0, 2.0, 3.0, 3.0]

model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="SpUNet-v1m1",
        in_channels=7,
        num_classes=9,
        base_channels=32,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 2, 2, 2, 2, 2, 2, 2),
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1, weight=class_weights),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=2.0, ignore_index=-1),
    ],
)

# Regularisierung
optimizer = dict(type="AdamW", lr=0.0005, weight_decay=0.2)
scheduler = dict(type="OneCycleLR", max_lr=0.0005, pct_start=0.04, anneal_strategy="cos", div_factor=10.0, final_div_factor=10000.0)

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
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            
            # STÄRKERES Scaling (0.8 bis 1.2 statt 0.9 bis 1.1)
            dict(type="RandomScale", scale=[0.6, 1.4]),
            
            dict(type="RandomFlip", p=0.5),
            
            # STÄRKERES Rauschen (RandomDrop entfernt, dafür Jitter hoch)
            dict(type="RandomJitter", sigma=0.03, clip=0.10), 
            
            dict(type="GridSample", grid_size=0.02, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="SphereCrop", point_max=60000, mode="random"),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "grid_coord", "segment"), feat_keys=("color", "normal", "strength")),
        ],
        test_mode=False,
        loop=10, 
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="GridSample", grid_size=0.02, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "grid_coord", "segment"), feat_keys=("color", "normal", "strength")),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[dict(type="CenterShift", apply_z=True), dict(type="ToTensor")],
        test_mode=True,
    ),
)
