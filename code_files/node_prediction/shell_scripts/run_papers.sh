
python ../main_batch.py --method SAGMM --dataset ogbn-papers100M --metric acc --lr 0.001 --hidden_channels 256 \
    --gnn_num_layers 3 --gnn_dropout 0.3 --gnn_weight_decay 0.0005 --gnn_use_residual --gnn_use_weight  --gnn_use_bn --gnn_use_init --gnn_use_act \
    --batch_size 8192 --seed 123 --runs 10 --epochs 150 --eval_step 5 --device 2 --data_dir ../../dataset/ \
    --num_experts 4 --max_experts 4 --k 4 --importance_threshold_factor 0.6  \
    --encode_mode lap --agg_mode concat --folder_name temp \
    --runs_path ../../../runs --prune_interval 6 \
    --gate_method attention --prune_type new_logits --gate_type zeros \
    --add_imp_loss --add_diversity_loss --div_weight 0.05 --imp_weight 0.05 --add_sigmoid
