nohup python main.py  test \
    --config config.yaml \
    --data.dataset_dir datasets/nips \
    --data.source_modality t1 \
    --data.target_modality t2 \
    --data.test_batch_size 1 \
    --model.eval_mask False \
    --model.eval_subject False \
    --ckpt_path checkpoints/ixi_t1_t2.ckpt \
    > logs/test_nips_orig 2>&1 &
        # --data.dataset_dir datasets/nips/infant \
    # --ckpt_path logs/experiment/version_4/checkpoints/epoch=49-step=8700.ckpt \
