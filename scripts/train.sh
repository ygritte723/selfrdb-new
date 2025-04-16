nohup python main.py  fit \
    --config config.yaml \
    --data.dataset_dir datasets/infant \
    --data.source_modality t1 \
    --data.target_modality t2 \
    --ckpt_path logs/experiment/version_4/checkpoints/epoch=29-step=5220.ckpt
    # --data.train_batch_size 20 \
    # --data.val_batch_size 5 \