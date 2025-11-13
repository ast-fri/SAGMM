
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from gnn_models import GCN, GAT, GraphSAGE, SGC, GIN, ChebNet, GCNwithJK, MixHopNet



class GMoE_network_noisy(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """
    
    def __init__(self, args, n, input_size, hid_size, num_classes, num_layers, dropout, num_experts,
                 noisy_gating=True, k=4, coef=1e-2):
        super(GMoE_network_noisy, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.k = 4
        self.args = args
        self.num_nodes = n
        self.loss_coef = coef
        self.graph_weight = args.graph_weight
        self.aggregate = args.aggregate
        self.experts_pool = [
            GCN(in_channels=input_size, hidden_channels=hid_size, num_layers=num_layers, dropout=dropout, use_bn=args.gnn_use_bn, use_ln=args.gnn_use_ln),
            GAT(in_channels=input_size, hidden_channels=hid_size, num_layers=num_layers, dropout=dropout, use_bn=args.gnn_use_bn, use_ln=args.gnn_use_ln),
            GraphSAGE(in_channels=input_size, hidden_channels=hid_size, num_layers=num_layers, dropout=dropout, use_bn=args.gnn_use_bn, use_ln=args.gnn_use_ln),
            SGC(in_channels=input_size, hidden_channels=hid_size, num_layers=num_layers, dropout=dropout, use_bn=args.gnn_use_bn, use_ln=args.gnn_use_ln),
            GCNwithJK(in_channels=input_size, hidden_channels=hid_size, num_layers=num_layers, dropout=dropout, use_bn=args.gnn_use_bn, use_ln=args.gnn_use_ln),
            ChebNet(in_channels=input_size, hidden_channels=hid_size, num_layers=num_layers, dropout=dropout, use_bn=args.gnn_use_bn, use_ln=args.gnn_use_ln),
            GIN(in_channels=input_size, hidden_channels=hid_size, num_layers=num_layers, dropout=dropout, use_bn=args.gnn_use_bn, use_ln=args.gnn_use_ln),
            MixHopNet(in_channels=input_size, hidden_channels=hid_size, num_layers=num_layers, dropout=dropout, use_bn=args.gnn_use_bn, use_ln=args.gnn_use_ln)
        ]
        self.experts = nn.ModuleList()
        ## for counting the distribution of nodes
        # Assign names to experts in the pool
        self.experts_pool_names = ['GCN', 'GAT', 'GraphSAGE', 'SGC', 'GCNwithJK', 'ChebNet', 'GIN', 'MixhopNet']
        

        # Directly initialize experts
        self.experts = nn.ModuleList([
            self.experts_pool[idx] for idx in range(num_experts)
        ])

        # Initialize expert count tensor of size max_expert_num (full expert list)
        self.expert_counts = torch.zeros(self.args.max_experts, device=args.device, requires_grad=False)
        self.gating_scores = torch.zeros(size=(args.max_experts,), requires_grad=False)
        # self.experts = nn.ModuleList([
        #     self.experts_pool[idx] for idx in range(num_experts) if idx != self.exclude_expert
        # ])
        self.expert_gate_indices = [idx for idx in range(num_experts)]
        self.expert_names = [self.experts_pool_names[idx] for idx in range(num_experts)]
        self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert(self.k <= self.num_experts)

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

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    # def forward(self, x, edge_index, edge_attr=None):
    def forward(self, x, edge_index, x_agg = None):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        gates, load = self.noisy_top_k_gating(x, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        #
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= self.loss_coef
        # loss = 0
        print_val = False
        if(self.args.dataset == "ogbn-arxiv" or self.args.dataset == "ogbn-proteins"):
            print_val = True
            print("Gates: ", gates)
            print("W gate: ", self.w_gate)
        expert_outputs = []
        for i in range(self.num_experts):
            expert_i_output = self.experts[i](x, edge_index)
            expert_outputs.append(expert_i_output)
        expert_outputs = torch.stack(expert_outputs, dim=1) # shape=[num_nodes, num_experts, d_feature]

        # gates: shape=[num_nodes, num_experts]
        y = gates.unsqueeze(dim=-1) * expert_outputs
        y = y.mean(dim=1)

        return y, loss, 0
