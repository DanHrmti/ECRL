from argparse import Namespace

training_params = Namespace(
    model_type="sa",  # ["sa", "slate"]
    dataset="panda",
    num_slots=10,
    num_iterations=2,  # 3
    accumulate_grad_batches=1,
    data_root="<path>",
    accelerator="gpu",
    devices=-1,
    max_steps=-1,
    num_sanity_val_steps=1,
    num_workers=4,
    is_logger_enabled=True,
    gradient_clip_val=0.05,  # 0.0
    n_samples=8,
)

slot_attention_params = Namespace(
    lr_main=4e-4,
    batch_size=32,
    val_batch_size=8,
    resolution=(128, 128),
    slot_size=64,
    max_epochs=1000,
    max_steps=500_000,
    weight_decay=0.0,
    mlp_hidden_size=256,  # 128
    scheduler="warmup_and_decay",
    scheduler_gamma=0.5,
    warmup_steps_pct=0.02,
    decay_steps_pct=0.2,
    use_separation_loss=False,  # "entropy"
    separation_tau_start=60_000,
    separation_tau_end=65_000,
    separation_tau_max_val=0.003,
    separation_tau=None,
    use_area_loss=False,  # True
    area_tau_start=60_000,
    area_tau_end=65_000,
    area_tau_max_val=0.006,
    area_tau=None,
)

slate_params = Namespace(
    lr_dvae=3e-4,
    lr_main=1e-4,
    weight_decay=0.0,
    batch_size=32,  # 50
    val_batch_size=8,  # 50
    max_epochs=1000,
    patience=8,
    gradient_clip_val=0.05,  # 1.0,
    resolution=(128, 128),
    num_dec_blocks=4,  # 8
    vocab_size=4096,
    d_model=192,
    num_heads=4,  # 8
    dropout=0.1,
    slot_size=192,
    mlp_hidden_size=192,
    tau_start=1.0,
    tau_final=0.1,
    tau_steps=30000,
    scheduler="warmup",
    lr_warmup_steps=30000,
    hard=False,
)

def merge_namespaces(one: Namespace, two: Namespace):
    return Namespace(**{**vars(one), **vars(two)})
