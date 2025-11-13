
python ../main.py --method SAGMM --dataset ogbl-ppa --metric acc --lr 0.01 --hidden_channels 256 \
    --gnn_num_layers 3  --gnn_dropout 0.0 --gnn_weight_decay 0. --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_act  \
    --seed 42 --runs 5 --epochs 101 --eval_step 5 --device 0 --data_dir ../../dataset/ \
    --num_experts 4  --k 1 --max_experts 4 --folder_name temp --batch_size 65536 \
    --importance_threshold_factor 0.5 --coef 0.001 \
    --runs_path ../../../runs  --prune_interval 24 \
    --gate_method attention --prune_type new_logits --gate_type zeros \
    --add_imp_loss --add_diversity_loss --encode_mode multihop_lap --agg_mode concat --add_sigmoid  --add_aux_loss \
    --imp_weight 0.05  --div_weight 0.05 --add_expert_norm
    
    
    