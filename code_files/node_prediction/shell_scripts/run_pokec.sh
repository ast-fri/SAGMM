
python ../main_batch.py --method SAGMM  --dataset pokec --rand_split --metric acc --lr 0.01 --hidden_channels 64 \
 --gnn_num_layers 3 --gnn_dropout 0. --gnn_weight_decay 0. --gnn_use_residual --gnn_use_weight --gnn_use_bn --gnn_use_init --gnn_use_act \
 --batch_size 500000 --seed 123 --runs 10 --epochs 1000 --eval_step 30 --device 1 --data_dir ../../dataset/ \
 --num_experts 8 --k 4 --importance_threshold_factor 0.5 --max_experts 8 \
 --encode_mode multihop_lap --agg_mode concat --folder_name temp  \
 --runs_path ../../../runs --prune_interval 20 \
 --gate_method attention --prune_type new_logits --gate_type zeros \
 --add_sigmoid --add_imp_loss --add_diversity_loss \
 --imp_weight 0.1 --div_weight 0.1
