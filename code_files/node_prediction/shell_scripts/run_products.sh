
python ../main_batch.py --method SAGMM --dataset ogbn-products --metric acc --lr 0.01 --hidden_channels 256 \
    --gnn_num_layers 3 --gnn_dropout 0.5 --gnn_weight_decay 0. --gnn_use_residual --gnn_use_weight --gnn_use_ln --gnn_use_init --gnn_use_act  \
    --batch_size 8192 --seed 123 --runs 10 --epochs 500 --eval_step 40 --device 3 --data_dir ../../dataset/ \
    --num_experts 4 --k 4 --importance_threshold_factor 0.6 --max_experts 4 \
    --encode_mode multihop_lap --agg_mode concat --folder_name temp \
    --runs_path ../../../runs --prune_interval 48 \
    --gate_method attention --prune_type new_logits --gate_type zeros \
    --add_imp_loss  --add_sigmoid  --imp_weight 0.5 --div_weight 0.05 --fan_out 10,5,2 

