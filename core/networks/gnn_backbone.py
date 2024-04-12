import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
# from core.networks.misc import (
#     ParallelLinear,
#     init_volume_scale,
# )

class ParallelLinear(nn.Module):

    def __init__(self, n_parallel, in_feat, out_feat, share=False, bias=True):
        super().__init__()
        self.n_parallel = n_parallel
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.share = share

        if not self.share:
            self.register_parameter('weight',
                                    nn.Parameter(torch.randn(n_parallel, in_feat, out_feat),
                                                 requires_grad=True)
                                   )
            if bias:
                self.register_parameter('bias',
                                        nn.Parameter(torch.randn(1, n_parallel, out_feat),
                                                     requires_grad=True)
                                       )
        else:
            self.register_parameter('weight', nn.Parameter(torch.randn(1, in_feat, out_feat),
                                                           requires_grad=True))
            if bias:
                self.register_parameter('bias', nn.Parameter(torch.randn(1, 1, out_feat), requires_grad=True))
        if not hasattr(self, 'bias'):
            self.bias = None
        #self.bias = nn.Parameter(torch.Tensor(n_parallel, 1, out_feat))
        self.reset_parameters()
        """
        self.conv = nn.Conv1d(in_feat * n_parallel, out_feat * n_parallel,
                              kernel_size=1, groups=n_parallel, bias=bias)
        """

    def reset_parameters(self):

        for n in range(self.n_parallel):
            # transpose because the weight order is different from nn.Linear
            nn.init.kaiming_uniform_(self.weight[n].T.data, a=math.sqrt(5))

        if self.bias is not None:
            #fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0].T)
            #bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            #nn.init.uniform_(self.bias, -bound, bound)
            nn.init.constant_(self.bias.data, 0.)

    def forward(self, x):
        weight, bias = self.weight, self.bias
        if self.share:
            weight = weight.expand(self.n_parallel, -1, -1)
            if bias is not None:
                bias = bias.expand(-1, self.n_parallel, -1)
        out = torch.einsum("bkl,klj->bkj", x, weight.to(x.device))
        if bias is not None:
            out = out + bias.to(x.device)
        return out

    def extra_repr(self):
        return "n_parallel={}, in_features={}, out_features={}, bias={}".format(
            self.n_parallel, self.in_feat, self.out_feat, self.bias is not None
        )

def init_volume_scale(base_scale, skel_profile, skel_type):
    # TODO: hard-coded some parts for now ...
    # TODO: deal with multi-subject
    joint_names = skel_type.joint_names
    N_joints = len(joint_names)
    bone_lens = skel_profile['bone_lens'][0]
    bone_lens_to_child = skel_profile['bone_lens_to_child'][0]

    # indices to all body parts
    head_idxs = skel_profile['head_idxs']
    torso_idxs = skel_profile['torso_idxs']
    arm_idxs = skel_profile['arm_idxs']
    leg_idxs = skel_profile['leg_idxs']
    collar_idxs = skel_profile['collar_idxs']

    # some widths
    shoulder_width = skel_profile['shoulder_width'][0]
    knee_width = skel_profile['knee_width'][0]
    collar_width = skel_profile['knee_width'][0]

    # init the scale for x, y and z
    # width, depth
    x_lens = torch.ones(N_joints) * base_scale
    y_lens = torch.ones(N_joints) * base_scale

    # half-width of thighs cannot be wider than the distant between knees in rest pose
    x_lens[leg_idxs] = knee_width * 0.5
    y_lens[leg_idxs] = knee_width * 0.5

    #  half-width of your body and head cannot be wider than shoulder distance (to some scale)
    #x_lens[torso_idxs] = shoulder_width * 0.70
    #y_lens[torso_idxs] = shoulder_width * 0.70
    x_lens[torso_idxs] = shoulder_width * 0.50
    y_lens[torso_idxs] = shoulder_width * 0.50
    x_lens[collar_idxs] = collar_width * 0.40
    y_lens[collar_idxs] = collar_width * 0.40

    #  half-width of your arms cannot be wider than collar distance (to some scale)
    x_lens[arm_idxs] = collar_width * 0.40
    y_lens[arm_idxs] = collar_width * 0.40

    # set scale along the bone direction
    # don't need full bone lens because the volume is supposed to centered at the middle of a bone
    z_lens = torch.tensor(bone_lens_to_child.copy().astype(np.float32))
    z_lens = z_lens * 0.8

    # deal with end effectors: make them grow freely
    """
    z_lens[z_lens < 0] = z_lens.max()
    # give more space to head as we do not have head-top joint
    z_lens[head_idxs] = z_lens.max() * 1.1 
    """
    x_lens[head_idxs] = shoulder_width * 0.30
    y_lens[head_idxs] = shoulder_width * 0.35
    # TODO: hack: assume at index 1 we have the head
    y_lens[head_idxs[1]] = shoulder_width * 0.6
    z_lens[head_idxs] = z_lens.max() * 0.30

    # find the lengths from end effector to their parents
    end_effectors = np.array([i for i, v in enumerate(z_lens) if v < 0 and i not in head_idxs])
    z_lens[end_effectors] = torch.tensor(skel_profile['bone_lens_to_child'][0][skel_type.joint_trees[end_effectors]].astype(np.float32))

    # handle hands and foots
    scale = torch.stack([x_lens, y_lens, z_lens], dim=-1)

    return scale

'''
Modified from Skeleton-aware Networks https://github.com/DeepMotionEditing/deep-motion-editing
'''
def skeleton_to_graph(skel=None, edges=None):
    ''' Turn skeleton definition to adjacency matrix and edge list
    '''

    if skel is not None:
        edges = []
        for i, j in enumerate(skel.joint_trees):
            if i == j:
                continue
            edges.append([j, i])
    else:
        assert edges is not None

    n_nodes = np.max(edges) + 1
    adj = np.eye(n_nodes, dtype=np.float32)

    for edge in edges:
        adj[edge[0], edge[1]] = 1.0
        adj[edge[1], edge[0]] = 1.0

    return adj, edges

class DenseWGCN(nn.Module):
    """ Basic GNN layer with learnable adj weights
    """
    def __init__(
            self, 
            adj, 
            in_channels, 
            out_channels, 
            init_adj_w=0.05, 
            bias=True, 
            **kwargs
    ):
        super(DenseWGCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        adj = adj.clone()
        idx = torch.arange(adj.shape[-1])
        adj[:, idx, idx] = 1

        init_w = init_adj_w
        perturb = 0.1
        adj_w = (adj.clone() * (init_w + (torch.rand_like(adj) - 0.5 ) * perturb).clamp(min=0.01, max=1.0))
        adj_w[:, idx, idx] = 1.0


        self.lin = nn.Linear(in_channels, out_channels)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=True)
        else:
            self.register_parameter('bias', None)

        self.register_buffer('adj', adj) # fixed, not learnable
        self.register_parameter('adj_w', nn.Parameter(adj_w, requires_grad=True)) # learnable

    def get_adjw(self):
        adj, adj_w = self.adj, self.adj_w

        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        adj_w = adj_w.unsqueeze(0) if adj_w.dim() == 2 else adj_w
        adj_w = adj_w * adj # masked out not connected part

        return adj_w

    def forward(self, x):

        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj_w = self.get_adjw().to(x.device)

        out = self.lin(x)
        out = torch.matmul(adj_w, out)

        if self.bias is not None:
            out = out + self.bias

        return out

class DensePNGCN(DenseWGCN):
    """ Basic GNN layer with learnable adj weights, and each node has its own linear layer
    """
    def __init__(
        self, 
        adj, 
        in_channel, 
        out_channel,
        *args, 
        **kwargs
    ):
        super(DensePNGCN, self).__init__(
            adj, 
            in_channel, 
            out_channel,
            *args, 
            **kwargs
        )
        self.lin = ParallelLinear(adj.shape[-1], in_channel, out_channel, bias=False)

class BasicGNN(nn.Module):
    """ A basic GNN with several graph layers
    """

    def __init__(
            self, 
            skel_type,
            per_node_input, 
            output_ch,
            W=64, 
            D=4,
            gcn_module=DensePNGCN, 
            gcn_module_kwargs={}, 
            nl=F.relu, 
            rigid_idxs=None,
            mask_root=True,
            skel_profile=None,
        ):
        """
        mask_root: Bool, to remove root input so everything is in relative coord
        """
        super(BasicGNN, self).__init__()

        self.skel_type = skel_type
        self.per_node_input = per_node_input
        self.output_ch = output_ch
        self.skel_profile = skel_profile

        self.rigid_idxs = rigid_idxs
        self.mask_root = mask_root
        self.W = W
        self.D = D

        adj_matrix, _ = skeleton_to_graph(skel_type)
        self.adj_matrix = adj_matrix
        self.gcn_module_kwargs = gcn_module_kwargs

        self.nl = nl
        if output_ch is None:
            self.output_ch = self.W + 1
        else:
            self.output_ch = output_ch
        self.init_network(gcn_module)

    def init_network(self, gcn_module):
        adj_matrix, skel_type = self.adj_matrix, self.skel_type
        per_node_input = self.per_node_input
        W, D = self.W, self.D

        n_nodes = adj_matrix.shape[-1]
        adj_matrix = torch.tensor(adj_matrix).view(1, n_nodes, n_nodes)

        layers = [gcn_module(adj_matrix, per_node_input, W, **self.gcn_module_kwargs)]
        for i in range(D-2):
            layers += [gcn_module(adj_matrix, W, W, **self.gcn_module_kwargs)]

        layers += [gcn_module(adj_matrix, W, self.output_ch, **self.gcn_module_kwargs)]
        self.layers = nn.ModuleList(layers)

        if self.mask_root:
            # mask root inputs, so that everything is in relative coordinate
            mask = torch.ones(1, len(self.skel_type.joint_names), 1)
            mask[:, self.skel_type.root_id, :] = 0.
            self.register_buffer('mask', mask)

    def forward(self, inputs, **kwargs):

        n = inputs
        if self.mask_root:
            n = n * self.mask

        for i, l in enumerate(self.layers):
            n = l(n)
            if (i + 1) < len(self.layers) and self.nl is not None:
                n = self.nl(n, inplace=True)
            if (i + 2) == len(self.layers) and self.rigid_idxs is not None:
                n = n[:, self.rigid_idxs]
        return n 

    def get_adjw(self):
        adjw_list = []

        for m in self.modules():
            if hasattr(m, 'adj_w'):
                adjw_list.append(m.get_adjw())

        return adjw_list

class GNN_Sine(BasicGNN):
    def __init__(
            self,
            *args,
            sin_w0=2.,
            voxel_feat=1,
            fc_D=2,
            fc_module=ParallelLinear,
            opt_scale=False,
            base_scale=0.5,
            alpha=2.,
            beta=6.,
            **kwargs
    ):
        self.sin_w0 = sin_w0
        self.fc_module = fc_module
        self.voxel_feat = voxel_feat
        self.fc_D = fc_D
        self.opt_scale = opt_scale
        self.alpha = alpha
        self.beta = beta
        self.voxel_feat_per_axis = self.voxel_feat // 3

        super(GNN_Sine, self).__init__(*args,
                                        **kwargs)

        self.init_scale(base_scale)

    @property
    def output_size(self):
        return self.voxel_feat

    def init_scale(self, base_scale):
        N_joints = len(self.skel_type.joint_names)

        scale = torch.ones(N_joints, 3) * base_scale
        if self.skel_profile is not None:
            scale = init_volume_scale(base_scale, self.skel_profile, self.skel_type)
        self.register_buffer('base_scale', scale.clone())
        self.register_parameter('axis_scale', nn.Parameter(scale, requires_grad=self.opt_scale))

    def get_axis_scale(self):
        axis_scale = self.axis_scale.abs()
        diff = axis_scale.detach() - self.base_scale * 0.95
        return torch.maximum(axis_scale, axis_scale - diff)

    def check_invalid(self, x):
        """ Assume points are in volume space already
        Args
        ----
        x: tensor [N_rays, N_samples, N_joints, 3]
        """
        x_v = x / self.get_axis_scale().reshape(1, 1, -1, 3).abs()
        invalid = ((x_v.abs() > 1).sum(-1) > 0).float()
        return x_v, invalid

    def init_network(self, gcn_module):
        adj_matrix, skel_type = self.adj_matrix, self.skel_type
        per_node_input = self.per_node_input
        W, D, fc_D = self.W, self.D, self.fc_D

        n_nodes = adj_matrix.shape[-1]
        adj_matrix = torch.tensor(adj_matrix).view(1, n_nodes, n_nodes)

        layers = [gcn_module(adj_matrix, per_node_input, W, **self.gcn_module_kwargs)]
        for i in range(D - 2):
            if i + 1 < fc_D:
                layers += [gcn_module(adj_matrix, W, W, **self.gcn_module_kwargs)]
            else:
                layers += [self.fc_module(n_nodes, W, W)]

        if self.fc_module in [ParallelLinear]:
            n_nodes = len(self.rigid_idxs) if self.rigid_idxs is not None else n_nodes
            layers += [self.fc_module(n_nodes, W, self.output_size)]
        else:
            layers += [self.fc_module(adj_matrix, W, self.output_size, **self.gcn_module_kwargs)]

        if self.mask_root:
            mask = torch.ones(1, len(self.skel_type.joint_names), 1)
            mask[:, self.skel_type.root_id, :] = 0.
            self.register_buffer('mask', mask)

        self.layers = nn.ModuleList(layers)
        self.layers.apply(self.frequency_init(self.sin_w0))
        self.layers[0].apply(self.first_layer_sine_init)

    def frequency_init(self, freq):
        def init(m):
            with torch.no_grad():
                if isinstance(m, ParallelLinear):
                    num_input = m.weight.size(-1)
                    m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)

        return init

    def first_layer_sine_init(self, m):
        with torch.no_grad():
            if isinstance(m, ParallelLinear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-1.0 / num_input, 1.0 / num_input)

    def forward(self, inputs, **kwargs):

        n = inputs
        if self.mask_root:
            n = n * self.mask

        for i, l in enumerate(self.layers):
            n = l(n)
            n = torch.sin(n * self.sin_w0)
        return n

