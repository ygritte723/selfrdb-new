nohup python main.py  fit \
    --config config.yaml \
    --data.dataset_dir datasets/nips \
    --data.source_modality t1 \
    --data.target_modality t2 \
    --trainer.log_every_n_steps 50 \
    --trainer.enable_progress_bar false \
  > logs/train_nips.log 2>&1 &
    # --ckpt_path logs/experiment/version_4/checkpoints/epoch=29-step=5220.ckpt
    # --data.train_batch_size 20 \
    # --data.val_batch_size 5 \