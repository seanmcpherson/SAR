# Copyright (c) 2022 Intel Corporation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import List, Union, Dict
from argparse import ArgumentParser
import os
import logging
import time
import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
import dgl  # type: ignore

import sar


parser = ArgumentParser(
    description="GNN training on node classification tasks in homogeneous graphs")


parser.add_argument(
    "--partitioning-json-file",
    type=str,
    default="",
    help="Path to the .json file containing partitioning information "
)

parser.add_argument('--ip-file', default='./ip_file', type=str,
                    help='File with ip-address. Worker 0 creates this file and all others read it ')


parser.add_argument('--backend', default='nccl', type=str, choices=['ccl', 'nccl', 'mpi'],
                    help='Communication backend to use '
                    )

parser.add_argument(
    "--cpu-run", action="store_true",
    help="Run on CPUs if set, otherwise run on GPUs "
)


parser.add_argument('--train-iters', default=100, type=int,
                    help='number of training iterations ')

parser.add_argument(
    "--lr",
    type=float,
    default=1e-2,
    help="learning rate"
)


parser.add_argument('--rank', default=0, type=int,
                    help='Rank of the current worker ')

parser.add_argument('--world-size', default=2, type=int,
                    help='Number of workers ')

parser.add_argument('--hidden-layer-dim', default=256, type=int,
                    help='Dimension of GNN hidden layer')

parser.add_argument('--single-node', action='store_true', 
                    help='Run distributed on a single node' )


class GNNModel(nn.Module):
    def __init__(self,  in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()

        self.convs = nn.ModuleList([
            # pylint: disable=no-member
            dgl.nn.SAGEConv(in_dim, hidden_dim, aggregator_type='mean'),
            # pylint: disable=no-member
            dgl.nn.SAGEConv(hidden_dim, hidden_dim, aggregator_type='mean'),
            # pylint: disable=no-member
            dgl.nn.SAGEConv(hidden_dim, out_dim, aggregator_type='mean'),
        ])

    def forward(self,  graph: sar.GraphShardManager, features: torch.Tensor):
        for idx, conv in enumerate(self.convs):
            features = conv(graph, features)
            if idx < len(self.convs) - 1:
                features = F.relu(features, inplace=True)

        return features

pointer_list = []

class PointerTensorObj(object):

    def __init__(self, data, pointer=[], **kwargs):
        self._t = torch.as_tensor(data, **kwargs)
        self._pointer = pointer
        self._pointer.append(self)

    #def __repr__(self):
    #    return "Metadata:\n{}\n\ndata:\n{}".format(self._metadata, self._t)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        pointers = tuple(a._pointer for a in args if hasattr(a, '_pointer'))
        args = [a._t if hasattr(a, '_t') else a for a in args]
        assert len(pointers) > 0
        ret = func(*args, **kwargs)
        return PointerTensorObj(ret, pointer=pointers[0])

INPLACE_FUNCTIONS = [
    torch.Tensor.resize_,
    torch.Tensor.copy_,
    torch.Tensor.storage
]

class PointerTensor(torch.Tensor):
    # Is data even needed? 
    def __init__(self, data, pointer=[], func="", **kwargs):
        self._pointer = pointer
        self._pointer.append(self)
        self._func = func
        
    @staticmethod
    def __new__(cls, x, pointer=[], *args, **kwargs):
        return super().__new__(cls, x, *args, **kwargs)
    
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        pointers = tuple(a._pointer for a in args if hasattr(a, '_pointer'))
        if len(pointers) == 0:
            pointers = [[]]
        #import ipdb; ipdb.set_trace()
        parent = super().__torch_function__(func, types, args, kwargs)
        if func not in INPLACE_FUNCTIONS and not hasattr(parent, '_pointer'):
            parent.__init__([], pointer=pointers[0], func=func)
        return parent
    
def main():
    args = parser.parse_args()
    print('args', args)
    
    use_gpu = torch.cuda.is_available() and not args.cpu_run
    device = torch.device('cuda' if use_gpu else 'cpu')

    # Obtain the ip address of the master through the network file system
    
    master_ip_address = sar.nfs_ip_init(args.rank, args.ip_file)
    if args.single_node:
        sar.initialize_single_node(args.rank,
                                   args.world_size, master_ip_address,
                                   device)
    else:
        sar.initialize_comms(args.rank,
                             args.world_size, master_ip_address,
                             args.backend)
    
    #import ipdb; ipdb.set_trace()
    pointer_list = []
    ttensor = PointerTensor([[1, 2], [3, 4]], pointer=pointer_list)
    t = torch.tensor([[1, 2], [1, 2]])
    added  = torch.add(t, ttensor)
    print(pointer_list)
    import ipdb; ipdb.set_trace()
    torch.save(torch.Tensor(ttensor), "ttensor.pt")
    tmp_tensor = torch.load("ttensor.pt")
    tttensor = PointerTensor(tmp_tensor)
    import ipdb; ipdb.set_trace()

    mask_list = []; labels_list = []; features_list = []
    num_labels = 0
    for rank_idx in range(args.world_size):
        # Load DGL partition data
        partition_data = sar.load_dgl_partition_data(
            args.partitioning_json_file, rank_idx, device)
        
        # Obtain train,validation, and test masks
        # These are stored as node features. Partitioning may prepend
        # the node type to the mask names. So we use the convenience function
        # suffix_key_lookup to look up the mask name while ignoring the
        # arbitrary node type
        masks = {}
        for mask_name, indices_name in zip(['train_mask', 'val_mask', 'test_mask'],
                                        ['train_indices', 'val_indices', 'test_indices']):
            boolean_mask = sar.suffix_key_lookup(partition_data.node_features,
                                                mask_name)
            masks[indices_name] = boolean_mask.nonzero(
                as_tuple=False).view(-1).to(device)

        labels = sar.suffix_key_lookup(partition_data.node_features,
                                    'labels').long().to(device)
        # Obtain the number of classes by finding the max label across all workers
        num_labels = max(num_labels, labels.max().item() + 1)
        #sar.comm.all_reduce(num_labels, dist.ReduceOp.MAX, move_to_comm_device=True)
        #num_labels = num_labels.item() 
        
        features = sar.suffix_key_lookup(partition_data.node_features, 'features').to(device)
        full_graph_manager = sar.construct_full_graph(partition_data).to(device)
        #torch.save(masks, 'masks.pt')
        #torch.save(labels, 'labels.pt')
        #torch.save(features, 'feats.pt')
        #torch.save(full_graph_manager, 'test.pt')
        import ipdb; ipdb.set_trace()
        features = PointerTensor(features, pointer=full_graph_manager.pointer_list)
        

        #We do not need the partition data anymore
        del partition_data
    
    gnn_model = GNNModel(features.size(1),
                         args.hidden_layer_dim,
                         num_labels).to(device)
    print('model', gnn_model)

    # Synchronize the model parmeters across all workers
    sar.sync_params(gnn_model)

    # Obtain the number of labeled nodes in the training
    # This will be needed to properly obtain a cross entropy loss
    # normalized by the number of training examples
    n_train_points = torch.LongTensor([masks['train_indices'].numel()])
    sar.comm.all_reduce(n_train_points, op=dist.ReduceOp.SUM, move_to_comm_device=True)
    n_train_points = n_train_points.item()

    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=args.lr)
    for train_iter_idx in range(args.train_iters):
        # Train
        t_1 = time.time()
        ipdb.set_trace()
        logits = gnn_model(full_graph_manager, features)
        loss = F.cross_entropy(logits[masks['train_indices']],
                               labels[masks['train_indices']], reduction='sum')/n_train_points

        optimizer.zero_grad()
        loss.backward()
        # Do not forget to gather the parameter gradients from all workers
        sar.gather_grads(gnn_model)
        optimizer.step()
        train_time = time.time() - t_1

        # Calculate accuracy for train/validation/test
        results = []
        for indices_name in ['train_indices', 'val_indices', 'test_indices']:
            n_correct = (logits[masks[indices_name]].argmax(1) ==
                         labels[masks[indices_name]]).float().sum()
            results.extend([n_correct, masks[indices_name].numel()])

        acc_vec = torch.FloatTensor(results)
        # Sum the n_correct, and number of mask elements across all workers
        sar.comm.all_reduce(acc_vec, op=dist.ReduceOp.SUM, move_to_comm_device=True)
        (train_acc, val_acc, test_acc) =  \
            (acc_vec[0] / acc_vec[1],
             acc_vec[2] / acc_vec[3],
             acc_vec[4] / acc_vec[5])
        ipdb.set_trace()
        result_message = (
            f"iteration [{train_iter_idx}/{args.train_iters}] | "
        )
        result_message += ', '.join([
            f"train loss={loss:.4f}, "
            f"Accuracy: "
            f"train={train_acc:.4f} "
            f"valid={val_acc:.4f} "
            f"test={test_acc:.4f} "
            f" | train time = {train_time} "
            f" |"
        ])
        print(result_message, flush=True)


if __name__ == '__main__':
    main()
