python ../main_batch.py --method SAGMM --rand_split --dataset deezer-europe  --lr 0.01 --gnn_num_layers 3 \
    --hidden_channels 96 --gnn_weight_decay 5e-05 --gnn_dropout 0.4 \
    --device 1 --data_dir ../../dataset/ \
    --epochs 1000 --runs 10 --batch_size 10000 --eval_step 20 \
    --num_experts 8  --max_experts 8 --encode_mode multihop_lap --agg_mode concat \
    --folder_name temp --importance_threshold_factor 0.4 \
    --runs_path ../../../runs  --prune_interval 60 \
    --gate_method attention --prune_type new_logits --gate_type zeros \
    --add_sigmoid --add_imp_loss --add_diversity_loss \
    --imp_weight 0.05 --div_weight 0.05
