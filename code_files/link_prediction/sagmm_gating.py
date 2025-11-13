import torch
import gc
import torch.nn as nn
from gnn_models import GCN, SAGE, GCNwithJK, SGC, GCNwithJK_O
import torch.nn.functional as F
from typing import  Any
from torch import Tensor

class SAGMMGateBackward(torch.autograd.Function):
    # jump the sign operation as the sign operation does not have gradients

    @staticmethod
    def forward(ctx: Any, scores: Tensor):
        signed_scores = torch.sign(scores)
        return signed_scores

    @staticmethod
    def backward(ctx:Any, grad_output: Tensor):
        return grad_output
    
class SAGMM_network(nn.Module):
    def __init__(self, args, n, input_size, hid_size, num_classes, num_layers, dropout, num_experts,
                 noisy_gating=True, k=4, coef=1e-2):
        super(SAGMM_network, self).__init__()
        self.input_size = input_size
        self.args = args
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.num_nodes = n
        self.device = args.device
        self.ema_decay = 0.9
        self.register_buffer("ema_contributions", torch.zeros(args.max_experts, device=args.device))


        
        if(args.dataset == "ogbl-ddi"):
            self.experts_pool = [
            GCN(in_channels=input_size, hidden_channels=hid_size, out_channels=num_classes, num_layers=num_layers, dropout=dropout, normalize=args.gnn_normalize, cached=args.gnn_cached),
            SAGE(in_channels=input_size, hidden_channels=hid_size, out_channels=num_classes, num_layers=num_layers, dropout=dropout),
            GCNwithJK_O(in_channels=input_size, hidden_channels=hid_size, out_channels=num_classes, num_layers=num_layers, dropout=dropout, use_bn=args.gnn_use_bn, use_ln=args.gnn_use_ln, normalize=args.gnn_normalize, cached=args.gnn_cached),
            SGC(in_channels=input_size, hidden_channels=hid_size, out_channels=num_classes, num_layers=num_layers, dropout=dropout, use_bn=args.gnn_use_bn, use_ln=args.gnn_use_ln),
        ]
        else:
            self.experts_pool = [
                GCN(in_channels=input_size, hidden_channels=hid_size, out_channels=num_classes, num_layers=num_layers, dropout=dropout, normalize=args.gnn_normalize, cached=args.gnn_cached),
                SAGE(in_channels=input_size, hidden_channels=hid_size, out_channels=num_classes, num_layers=num_layers, dropout=dropout),
                GCNwithJK(in_channels=input_size, hidden_channels=hid_size, out_channels=num_classes, num_layers=num_layers, dropout=dropout, use_bn=args.gnn_use_bn, use_ln=args.gnn_use_ln, normalize=args.gnn_normalize, cached=args.gnn_cached),
                SGC(in_channels=input_size, hidden_channels=hid_size, out_channels=num_classes, num_layers=num_layers, dropout=dropout, use_bn=args.gnn_use_bn, use_ln=args.gnn_use_ln),
            ]
        self.experts = nn.ModuleList()
        # Assign names to experts in the pool
        self.experts_pool_names = ['GCN', 'GraphSAGE', 'GCNwithJK', 'SGC']
    
        # Directly initialize experts
        self.experts = nn.ModuleList([
            self.experts_pool[idx] for idx in range(num_experts)
        ])

        self.expert_gate_indices = [idx for idx in range(num_experts)]
        self.expert_names = [self.experts_pool_names[idx] for idx in range(num_experts)]
        
        # Gate initialization
        if(args.gate_type == "zeros"):
            print("Initialized with 0")
            self.gate_threshold = torch.nn.Parameter(torch.zeros(size=(num_experts,)), requires_grad=True)
        else:
            print("Initialized with rand n")
            self.gate_threshold = torch.nn.Parameter(torch.randn(1, num_experts) * 0.1, requires_grad=True)
  
        # Initialization of weight tensors
        if(args.gate_method == "attention"):
            if(args.agg_mode == "add" or args.agg_mode == "none"):
                self.W_q = nn.Linear(input_size, num_experts, bias=False)
                self.W_k = nn.Linear(input_size, num_experts, bias=False)
                self.W_v = nn.Linear(input_size, num_experts, bias=False)
                
            else:
                self.W_q = nn.Linear(input_size + args.max_freqs, num_experts, bias=False)
                self.W_k = nn.Linear(input_size + args.max_freqs, num_experts, bias=False)
                self.W_v = nn.Linear(input_size + args.max_freqs, num_experts, bias=False)
                

        elif(args.gate_method == "noisy_top_any"):
            print("Noisy top any gating")
            if(args.agg_mode == "add" or args.agg_mode == "none"):
                self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
                self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
            else:
                self.w_gate = nn.Parameter(torch.zeros(input_size + args.max_freqs, num_experts ), requires_grad=True)
                self.w_noise = nn.Parameter(torch.zeros(input_size + args.max_freqs, num_experts ), requires_grad=True)
        
        
        if(self.args.add_proj):
            self.projection = nn.Linear(args.hidden_channels, args.hidden_channels, bias=True)
        
        
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_parameter('experts_mask', torch.nn.Parameter(torch.ones(size=(num_experts,)), requires_grad=False))

    
    def remove_expert(self, expert_index, optimizer=None):
        if expert_index >= len(self.experts):
            print("Expert index out of range.")
            return optimizer

        # Retrieve the expert to remove before setting it to None
        expert_to_remove = self.experts[expert_index]
        gate_index = self.expert_gate_indices[expert_index]

        # Mark the expert as inactive
        self.experts[expert_index] = None
        self.expert_names[expert_index] = None
        self.expert_gate_indices[expert_index] = None
        self.num_experts -= 1

        # Update the experts mask in the gate
        self.experts_mask[gate_index] = 0.0

        # Remove optimizer state for the expert's parameters
        if optimizer is not None:
            params_to_remove = set(expert_to_remove.parameters())
            for param_group in optimizer.param_groups:
                param_group['params'] = [p for p in param_group['params'] if p not in params_to_remove]
            for param in params_to_remove:
                if param in optimizer.state:
                    del optimizer.state[param]

        self.ema_contributions[gate_index] = float('-inf')
        torch.cuda.empty_cache()
        gc.collect()
        return optimizer

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def diverse_gate_loss(self, gates, expert_mask):
        sims = torch.matmul(F.normalize(gates, dim=0).T, F.normalize(gates, dim=0))
        targets = torch.eye(sims.shape[0]).to(sims.device)
        sim_mask = torch.matmul(expert_mask.unsqueeze(0).T, expert_mask.unsqueeze(0))
        sim_loss = torch.norm(sims * sim_mask - targets * sim_mask)
        # del sims, targets, sim_mask
        return sim_loss

    def update_ema_contributions(self,gates, expert_outputs, valid_indices):
        """
        Updates the EMA of expert contributions using the current batch's gate scores and outputs.
        """
        with torch.no_grad():
            # Compute contribution magnitudes
            # [num_samples, num_experts, input_dim] -> [num_experts]
            contribution_magnitudes = (gates.unsqueeze(dim=-1) * expert_outputs).sum(dim=0).norm(dim=-1)

            # Map valid indices to gate indices
            actual_gate_indices = torch.tensor(
                [self.expert_gate_indices[idx] for idx in valid_indices],
                device=self.device
            )

            # Mask for valid gate indices
            valid_mask = actual_gate_indices >= 0

            # Apply EMA updates only for valid experts
            valid_gate_indices = actual_gate_indices[valid_mask]
            valid_contributions = contribution_magnitudes[actual_gate_indices]

            # Update EMA contributions
            self.ema_contributions[valid_gate_indices] = (
                self.ema_contributions[valid_gate_indices] * (1 - self.ema_decay)
                + self.ema_decay * valid_contributions
            )

            # Clear intermediate tensors
            del contribution_magnitudes, actual_gate_indices, valid_gate_indices, valid_contributions

    def full_attention_conv(self, qs, ks, vs, output_attn=False):
        qs = qs / torch.norm(qs, p=2)  # [N, H, M]
        ks = ks / torch.norm(ks, p=2)  # [L, H, M]
        N = qs.shape[0]

        # Numerator
        kvs = torch.einsum("lhm,lhd->hmd", ks, vs)
        attention_num = torch.einsum("nhm,hmd->nhd", qs, kvs)  # [N, H, D]
        attention_num += N * vs

        # Denominator
        all_ones = torch.ones([ks.shape[0]]).to(ks.device)
        ks_sum = torch.einsum("lhm,l->hm", ks, all_ones)
        attention_normalizer = torch.einsum("nhm,hm->nh", qs, ks_sum)  # [N, H]

        # Attentive aggregated results
        attention_normalizer = torch.unsqueeze(
            attention_normalizer, len(attention_normalizer.shape))  # [N, H, 1]
        attention_normalizer += torch.ones_like(attention_normalizer) * N
        attn_output = attention_num / attention_normalizer  # [N, H, D]

        if output_attn:
            attention = torch.einsum("nhm,lhm->nlh", qs, ks).mean(dim=-1)  # [N, N]
            normalizer = attention_normalizer.squeeze(dim=-1).mean(dim=-1, keepdims=True)  # [N,1]
            attention = attention / normalizer
            return attn_output, attention

        return attn_output
 
    def forward(self, x, edge_index, x_agg = None, adj_t_norm = None, last_idx=False):
        valid_indices = [idx for idx, expert in enumerate(self.experts) if expert is not None]
        
        noise_epsilon=1e-2
        print_val = False
        if(x_agg == None):
            raise ValueError("Error in the code !!!!")
        
        if(self.args.gate_method == "attention"):
            Q = self.W_q(x_agg).unsqueeze(1)  # [N, 1, num_experts]
            K = self.W_k(x_agg).unsqueeze(1)  # [N, 1, num_experts]
            V = self.W_v(x_agg).unsqueeze(1)  # [N, 1, num_experts]
            
            attn_output = self.full_attention_conv(Q, K, V) # [N, 1, num_experts]
            attn_output = attn_output.squeeze(1)  # [N, num_experts]
        
            if(self.args.add_sigmoid):
                gate_scores_p = torch.sigmoid(attn_output)
            else:
                gate_scores_p = self.softplus(attn_output)
            
        
        elif(self.args.gate_method == "noisy_top_any"):
            clean_logits = x_agg @ self.w_gate
            if self.noisy_gating and self.training:
                raw_noise_stddev = x_agg @ self.w_noise
                noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
                noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
                w_logits = noisy_logits
            else:
                w_logits = clean_logits
           
            if(self.args.add_sigmoid):
                gate_scores_p = torch.sigmoid(w_logits)
            else:
                gate_scores_p = self.softplus(w_logits)


        gate_scores_p = gate_scores_p * self.experts_mask
        threshold = torch.sigmoid(self.gate_threshold) 
        new_logits = F.relu(gate_scores_p - threshold)
        new_logits = SAGMMGateBackward.apply(new_logits)
       
        no_expert_selected = new_logits.sum(dim=-1) == 0  
        
        if no_expert_selected.any():
            max_expert_idx = gate_scores_p[no_expert_selected].argmax(dim=-1)  # Compute only for relevant indices
            new_logits[no_expert_selected, max_expert_idx] = True  # Directly update 
        gate_scores = gate_scores_p * new_logits.float()  # Zero out non-selected experts
        
       
        output_shape = (x.shape[0], self.args.hidden_channels)
        expert_outputs = []
        for idx in range(len(self.experts)):
            if idx in valid_indices and self.experts[idx] is not None:
                if(self.args.dataset == "ogbl-ppa" and (idx == 0 or idx == 2)):
                    expert_output = self.experts[idx](x, adj_t_norm)
                else:
                    expert_output = self.experts[idx](x, edge_index)
                if(self.args.add_proj):
                    expert_output = self.projection(expert_output)
                if(self.args.add_expert_norm):
                    normalized_expert_output = F.normalize(expert_output, p=2, dim=-1)
                    expert_output = normalized_expert_output
            else:
                expert_output = torch.zeros(output_shape, device=x.device)
            expert_outputs.append(expert_output)
        expert_outputs = torch.stack(expert_outputs, dim=1)
       

        # Update EMA contributions during training
        if self.training:
            if(self.args.prune_type == "new_logits"):
                self.update_ema_contributions(new_logits, expert_outputs, valid_indices)
            else:
                self.update_ema_contributions(gate_scores, expert_outputs, valid_indices)
         
        y = torch.sum(gate_scores.unsqueeze(dim=-1) * expert_outputs, dim=1)

       
        loss = 0
        if(self.training):
            if(self.args.add_imp_loss):
                importance = gate_scores.sum(0)
                loss = loss + self.args.imp_weight * self.cv_squared(importance)
            
            if(self.args.add_diversity_loss):
                if(self.args.gate_method == "attention"):
                    loss = loss + self.args.div_weight * (self.diverse_gate_loss(self.W_q.weight.T, self.experts_mask))
                    loss = loss + self.args.div_weight * (self.diverse_gate_loss(self.W_k.weight.T, self.experts_mask))
                    loss = loss + self.args.div_weight * (self.diverse_gate_loss(self.W_v.weight.T, self.experts_mask))
                elif(self.args.gate_method == "noisy_top_any"):
                    loss = loss + self.args.div_weight * self.diverse_gate_loss(self.w_gate, self.experts_mask)
            

        # Print counts
        if(print_val and last_idx):
            with torch.no_grad():
                print("Training: ", self.training)
                print("Gate scores : ", gate_scores_p)
                print("Experts mask: ", self.experts_mask)
                print("threshold: ", threshold)
                print("Selected gate_scores after threshold: ",gate_scores)
                print("new logits", new_logits)
                top_k = torch.sum(new_logits > 0, dim=1).to(torch.int)
                counts = torch.bincount(top_k)
                print("top_k: ",top_k)
                ## print average and max number of k
                k_per_node = new_logits.sum(dim=-1) 
                avg_k = k_per_node.float().mean().item() 
                max_k = k_per_node.max().item() 
                print(f"Average k: {avg_k:.2f}, Max k: {max_k}")
                expert_norms = torch.norm(expert_outputs, dim=-1).mean(dim=0)  # [num_experts]
                print("Expert output L2 norms:", expert_norms)
                expert_means = expert_outputs.mean(dim=(0,2))  # [num_experts] 
                print("Expert output means:", expert_means)
                expert_stds = expert_outputs.std(dim=(0,2))   # [num_experts]
                print("Expert output stds:", expert_stds)
                for value, count in enumerate(counts):
                    if count > 0:  # Only print existing values
                        print(f"Value {value}: {count} occurrences")
                counts_per_expert = new_logits.sum(dim=0)

                
                print("Expert selection counts:")
                for i, count in enumerate(counts_per_expert):
                    if(count > 0):
                        print(f"  Expert {i}: {count.item()} nodes")

        if(print_val and self.training):
            log_info = (gate_scores, gate_scores_p, threshold, new_logits, avg_k, max_k, counts, counts_per_expert) 
        else:
            log_info = 0       
        return y, loss, log_info
    

    
    def reset_parameters(self):
        with torch.no_grad():
            # Reset parameters of experts in the pool
            for exp in self.experts_pool:
                exp.reset_parameters()

            # Reset the number of experts to the maximum
            self.num_experts = self.max_expert_num

            # Reinitialize experts
            self.experts = nn.ModuleList([
                self.experts_pool[idx] for idx in range(self.num_experts)
            ])
            self.expert_gate_indices = [idx for idx in range(self.num_experts)]
            self.expert_names = [self.experts_pool_names[idx] for idx in range(self.num_experts)]

            