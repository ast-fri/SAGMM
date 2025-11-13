## Example Usages, change the dataset name as per your requirement
python main.py --method SAGMM  \
 --dataset ogbg-molbbbp --lr 0.001 --hidden_channels 300 --num_layer 5 --drop_ratio 0.5  --gnn_use_bn \
 --batch_size 32 --seed 123 --runs 10 --epochs 200 --eval_step 10 --device 2 \
 --data_dir ../../dataset/ \
 --num_experts 4 --importance_threshold_factor 0.4 \
 --expert_type multiple_models \
 --folder_name temp \
 --runs_path ../runs/ \
 --prune_interval 40 \

#  python main.py --method SAGMM  \
#  --dataset ogbg-moltox21 --lr 0.001 --hidden_channels 300 --num_layer 5 --drop_ratio 0.5  --gnn_use_bn \
#  --batch_size 1000 --seed 123 --runs 10 --epochs 200 --eval_step 10 --device 2 \
#  --data_dir ../../dataset/ \
#  --num_experts 4 --importance_threshold_factor 0.4 \
#  --expert_type multiple_models \
#  --folder_name temp \
#  --runs_path runs/ \
#  --prune_interval 40 \