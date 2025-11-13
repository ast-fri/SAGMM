


python ../main_batch.py --method SAGMM --dataset ogbn-proteins --metric rocauc --lr 0.01 --hidden_channels 64 \
    --gnn_num_layers 3  --gnn_dropout 0. --gnn_weight_decay 0. --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_act \
    --batch_size 50000 --seed 123 --runs 10 --epochs 1000 --eval_step 40 --device 2 \
    --data_dir ../../dataset/  \
    --num_experts 8 --k 4  --encode_mode multihop_lap --agg_mode concat --folder_name temp --importance_threshold_factor 0.5 \
    --runs_path ../../../runs --max_experts 8 --prune_interval 38 \
    --gate_method attention --prune_type gate_scores --gate_type randn \
    --add_imp_loss --add_sigmoid 

