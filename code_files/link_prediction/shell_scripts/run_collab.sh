
python ../main.py --method SAGMM --dataset ogbl-collab --metric acc --lr 0.001 --hidden_channels 256 \
    --gnn_num_layers 3  --gnn_dropout 0.0 --gnn_weight_decay 0. --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_act  \
    --seed 42 --runs 10 --epochs 1000 --eval_step 1 --device 1 --data_dir ../../dataset/ \
    --num_experts 4  --k 1 --max_experts 4 --folder_name temp --batch_size 65536 \
    --importance_threshold_factor 0.3 --coef 0.001 \
    --runs_path ../../../runs --prune_interval 28 \
    --gate_method attention --prune_type new_logits --gate_type zeros \
    --encode_mode multihop_lap --agg_mode concat --add_sigmoid --add_aux_loss --add_imp_loss --add_diversity_loss \
    --imp_weight 0.1 --div_weight 0.05 --gnn_normalize --gnn_cached 
