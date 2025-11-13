

python ../main.py --method SAGMM  --rand_split --dataset yelp-chi --metric acc --lr 0.01 --hidden_channels 64  \
    --gnn_num_layers 3  --gnn_dropout 0.5 --gnn_weight_decay 0. --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_act \
    --seed 123 --runs 10 --epochs 1000 --eval_step 9 --device 0 --data_dir ../../dataset/ \
    --num_experts 8  --k 4 --encode_mode multihop_lap --agg_mode concat --folder_name temp \
    --importance_threshold_factor 0.5 \
    --runs_path ../../../runs --max_experts 8 --prune_interval 30 \
    --gate_method attention --prune_type new_logits --gate_type zeros \
    --add_diversity_loss --add_sigmoid --div_weight 0.1 
