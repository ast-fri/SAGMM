
python ../main.py --method SAGMM --dataset ogbl-ddi --metric acc --lr 0.005 --hidden_channels 256 \
    --gnn_num_layers 2  --gnn_dropout 0.5 --gnn_weight_decay 0. --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_act \
    --seed 123 --runs 10 --epochs 1000 --eval_step 9 --device 2 --data_dir ../../dataset/ \
    --num_experts 4  --k 1 --max_experts 4 --folder_name temp --batch_size 65536 --importance_threshold_factor 0.4 --coef 0.0005 \
    --runs_path ../../../runs  --prune_interval 48 \
    --gate_method attention --prune_type new_logits --gate_type zeros --gnn_normalize --gnn_cached \
    --encode_mode lap --agg_mode none --add_sigmoid --add_aux_loss --add_diversity_loss --add_imp_loss --imp_weight 0.05 --div_weight 0.05 
