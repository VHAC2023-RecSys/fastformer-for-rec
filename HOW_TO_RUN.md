
# Preprocess data

```bash
python data_generation.py
```

# Train model

```bash
python train.py --pretrained_model unilm --pretrained_model_path hf_models/unilm-base-cased --root_data_dir ./data/speedy_data/ --num_hidden_layers 8 --world_size 1 --lr 1e-4 --pretrain_lr 8e-6 --warmup True --schedule_step 240000 --warmup_step 1000 --batch_size 42 --npratio 4 --beta_for_cache 0.002 --max_step_in_cache 2 --savename speedyrec_mind --news_dim 256
```