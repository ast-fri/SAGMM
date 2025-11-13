
python ../main.py --method SAGMM --dataset ogbn-arxiv --metric acc --lr 0.001 --hidden_channels 128  \
    --gnn_num_layers 3  --gnn_dropout 0.5 --gnn_weight_decay 0. --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_act \
    --seed 123 --runs 10 --epochs 1000 --eval_step 9 --device 3 --data_dir ../../dataset/ \
    --num_experts 8  --max_experts 8 --k 1 --encode_mode multihop_lap --agg_mode concat --folder_name temp \
    --importance_threshold_factor 0.6  \
    --runs_path ../../../runs  --prune_interval 30 \
    --gate_method attention --prune_type new_logits --gate_type zeros \
    --add_sigmoid --add_imp_loss --add_diversity_loss --div_weight 0.05 --imp_weight 0.1
