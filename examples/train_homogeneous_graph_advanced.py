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


parser.add_argument('--log-level', default='INFO', type=str,
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                    help='SAR log level ')


parser.add_argument('--backend', default='ccl', type=str, choices=['ccl', 'nccl', 'mpi', 'gloo'],
                    help='Communication backend to use ')

parser.add_argument(
    "--construct-mfgs", action="store_true",
    help="Construct MFGs for all layers before training"
)

parser.add_argument(
    "--cpu-run", action="store_true",
    help="Run on CPUs if set, otherwise run on GPUs "
)

parser.add_argument('--train-mode', default='SAR',
                    type=str,
                    choices=['SAR', 'SA', 'one_shot_aggregation'],
                    help='Training mode to use: SAR (Sequential Aggregation and \
                    Rematerialization),SA (Sequential Aggregation), or one_shot_aggregation')


parser.add_argument('--train-iters', default=100, type=int,
                    help='number of training iterations ')

parser.add_argument('--max-collective-size', default=0, type=int,
                    help='The maximum allowed size of the data in a collective. \
If a collective would communicate more than this maximum, it is split into multiple collectives.\
Collective calls with large data may cause instabilities in some communication backends  ')

parser.add_argument(
    "--lr",
    type=float,
    default=1e-2,
    help="learning rate"
)

parser.add_argument('--gnn-layer', default='sage', type=str, choices=['gcn', 'sage', 'gat'],
                    help='GNN layer type')

parser.add_argument('--rank', default=-1, type=int,
                    help='Rank of the current worker ')

parser.add_argument('--world-size', default=-1, type=int,
                    help='Number of workers ')

parser.add_argument('--n-layers', default=3, type=int,
                    help='Number of GNN layers ')

parser.add_argument('--layer-dim', default=256, type=int,
                    help='Dimension of GNN hidden layer')


class GNNModel(nn.Module):
    def __init__(self,  gnn_layer: str, n_layers: int, layer_dim: int,
                 input_feature_dim: int, n_classes: int):
        super().__init__()

        assert n_layers >= 1, 'GNN must have at least one layer'
        dims = [input_feature_dim] + [layer_dim] * (n_layers-1) + [n_classes]

        self.convs = nn.ModuleList()
        for idx in range(len(dims) - 1):
            if gnn_layer == 'gat':
                # use 2 aattention heads
                layer = dgl.nn.GATConv(dims[idx], dims[idx+1], 2)  # pylint: disable=no-member
            elif gnn_layer == 'gcn':
                layer = dgl.nn.GraphConv(dims[idx], dims[idx+1])  # pylint: disable=no-member
            elif gnn_layer == 'sage':
                # Use mean aggregtion
                # pylint: disable=no-member
                layer = dgl.nn.SAGEConv(dims[idx], dims[idx+1],
                                        aggregator_type='mean')
            else:
                raise ValueError(f'unknown gnn layer type {gnn_layer}')
            self.convs.append(layer)

    def forward(self,  blocks: List[Union[sar.GraphShardManager, sar.DistributedBlock]],
                features: torch.Tensor):
        for idx, conv in enumerate(self.convs):
            features = conv(blocks[idx], features)
            if features.ndim == 3:  # GAT produces an extra n_heads dimension
                # collapse the n_heads dimension
                features = features.mean(1)

            if idx < len(self.convs) - 1:
                features = F.relu(features, inplace=True)

        return features


def infer_pass(gnn_model: torch.nn.Module,
               eval_blocks: List[Union[sar.GraphShardManager, sar.DistributedBlock]],
               features: torch.Tensor,
               masks: Dict[str, torch.Tensor],
               labels: torch.Tensor,
               mfg_blocks: bool):

    # If we had constructed MFGs, then the input nodes for the first block might be
    # a subset of the the nodes in the partition. Use the input_nodes member of
    # sar.GraphShardManager or sar.DistributedBlock to obtain the indices of the
    # input nodes to the first block and provide only the features of these nodes as input
    if mfg_blocks:
        features = features[eval_blocks[0].input_nodes]
    gnn_model.eval()
    with torch.no_grad():
        logits = gnn_model(eval_blocks, features)

    results = []
    if mfg_blocks:
        # The seed nodes for the eval MFGs are
        # torch.cat((masks['train_indices'],masks['val_indices'],masks['test_indices']))
        # These will be the nodes produced by the top layer MFG
        start_index = 0
        for indices_name in ['train_indices', 'val_indices', 'test_indices']:
            active_indices = masks[indices_name]
            active_logits = logits[start_index:  start_index + active_indices.numel()]
            if active_indices.numel() > 0:
                loss = F.cross_entropy(active_logits,
                                       labels[active_indices], reduction='sum')
                n_correct = (active_logits.argmax(1) ==
                             labels[active_indices]).float().sum()
                results.extend([loss.item(), n_correct.item(), active_indices.numel()])
                start_index += active_indices.numel()
            else:
                results.extend([0.0, 0.0, 0.0])
    else:  # No MFGs were constructed. We are using the full graph in each layer
        for indices_name in ['train_indices', 'val_indices', 'test_indices']:
            active_indices = masks[indices_name]
            active_logits = logits[active_indices]
            if active_indices.numel() > 0:
                loss = F.cross_entropy(active_logits,
                                       labels[active_indices], reduction='sum')
                n_correct = (active_logits.argmax(1) ==
                             labels[active_indices]).float().sum()
                results.extend([loss.item(), n_correct.item(), active_indices.numel()])
            else:
                results.extend([0.0, 0.0, 0.0])

    loss_acc_vec = torch.FloatTensor(results)
    # Sum the loss, n_correct, and number of mask elements across all workers
    sar.comm.all_reduce(loss_acc_vec, op=dist.ReduceOp.SUM, move_to_comm_device=True)

    (train_loss, train_acc, val_loss, val_acc, test_loss, test_acc) = \
        (loss_acc_vec[0] / loss_acc_vec[2],
         loss_acc_vec[1] / loss_acc_vec[2],
         loss_acc_vec[3] / loss_acc_vec[5],
         loss_acc_vec[4] / loss_acc_vec[5],
         loss_acc_vec[6] / loss_acc_vec[8],
         loss_acc_vec[7] / loss_acc_vec[8])

    return (train_loss, train_acc, val_loss, val_acc, test_loss, test_acc)


def train_pass(gnn_model: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               train_blocks: List[Union[sar.GraphShardManager, sar.DistributedBlock]],
               features: torch.Tensor,
               train_mask: torch.Tensor,
               labels: torch.Tensor,
               n_train_points: int,
               mfg_blocks: bool):

    # If we had constructed MFGs, then the input nodes for the first block might be
    # a subset of the the nodes in the partition. Use the input_nodes member of
    # sar.GraphShardManager or sar.DistributedBlock to obtain the indices of the
    # input nodes to the block and provide only the features of these nodes as input
    if mfg_blocks:
        features = features[train_blocks[0].input_nodes]

    gnn_model.train()
    logits = gnn_model(train_blocks, features)

    if mfg_blocks:
        # By construction, the output nodes of the top layer training MFG are
        # exactly the labeled nodes in the training set
        loss = F.cross_entropy(logits,
                               labels[train_mask], reduction='sum')/n_train_points
    else:
        loss = F.cross_entropy(logits[train_mask],
                               labels[train_mask], reduction='sum')/n_train_points

    optimizer.zero_grad()
    loss.backward()
    # Do not forget to gather the parameter gradients from all workers
    sar.gather_grads(gnn_model)
    optimizer.step()


def main():
    args = parser.parse_args()
    print('args', args)

    #Patch DGL's attention-based layers and RelGraphConv to support distributed graphs
    sar.patch_dgl()
    
    if args.rank == -1:
        # Try to infer the worker's rank from environment variables
        # created by mpirun or similar MPI launchers
        args.rank = int(os.environ.get("PMI_RANK", -1))
        if args.rank == -1:
            args.rank = int(os.environ["RANK"])

    if args.world_size == -1:
        # Try to infer the number of workers from environment variables
        # created by mpirun or similar launchers
        args.world_size = int(os.environ.get("PMI_SIZE", -1))
        if args.world_size == -1:
            args.world_size = int(os.environ["WORLD_SIZE"])

    use_gpu = torch.cuda.is_available() and not args.cpu_run

    device = torch.device('cuda' if use_gpu else 'cpu')
    if args.backend == 'nccl':
        comm_device = torch.device('cuda')
    else:
        comm_device = torch.device('cpu')

    sar.logging_setup(logging.getLevelName(args.log_level),
                      args.rank, args.world_size)

    # Obtain the ip address of the master through the network file system
    master_ip_address = sar.nfs_ip_init(args.rank, args.ip_file)
    sar.initialize_comms(args.rank,
                         args.world_size, master_ip_address,
                         args.backend, comm_device)

    # Load DGL partition data
    partition_data = sar.load_dgl_partition_data(
        args.partitioning_json_file, args.rank, device)

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
    num_labels = labels.max() + 1
    sar.comm.all_reduce(num_labels, dist.ReduceOp.MAX, move_to_comm_device=True)
    num_labels = num_labels.item()
    
    features = sar.suffix_key_lookup(partition_data.node_features, 'features').to(device)
    if args.construct_mfgs:
        # sar.construct_mfgs needs the global indices of the seed nodes.
        # We obtain the global indices by getting the indices of the labeled nodes
        # in the local partition and then adding the start node index for the local partition.
        # Global node indices are consecutive in each partition
        train_blocks = sar.construct_mfgs(partition_data,
                                          masks['train_indices'] +
                                          partition_data.node_ranges[sar.comm.rank()][0],
                                          args.n_layers)
        # During evaluation we want to also evaluate on the training nodes
        eval_blocks = sar.construct_mfgs(partition_data,
                                         torch.cat((masks['train_indices'],
                                                    masks['val_indices'],
                                                    masks['test_indices'])) +
                                         partition_data.node_ranges[sar.comm.rank()][0],
                                         args.n_layers)

        # If we use the one_shot_aggregation mode (mode 3), we need to use the
        # DistributedBlock representation instead of the GraphShardManager representation
        # The DistributedBlock representation can be obtained using get_full_partition_graph
        if args.train_mode == 'one_shot_aggregation':
            train_blocks = [block.get_full_partition_graph() for block in train_blocks]
            eval_blocks = [block.get_full_partition_graph() for block in eval_blocks]

        # Move the graph objects to the training device
        train_blocks = [block.to(device) for block in train_blocks]
        eval_blocks = [block.to(device) for block in eval_blocks]

    else:  # No MFGs. The same full graph in every layer
        full_graph_manager = sar.construct_full_graph(partition_data)
        if args.train_mode == 'one_shot_aggregation':
            full_graph_manager = full_graph_manager.get_full_partition_graph()
        full_graph_manager = full_graph_manager.to(device)
        train_blocks = [full_graph_manager] * args.n_layers
        eval_blocks = [full_graph_manager] * args.n_layers

    if args.train_mode == 'SA':
        # Only do sequential aggregation. Disable sequential rematerialization
        # of the computational graph in the backward pass. Will lead to higher
        # memory consumption
        sar.Config.disable_sr = True

    sar.Config.max_collective_size = args.max_collective_size

    #We do not need the partition data anymore
    del partition_data


    gnn_model = GNNModel(args.gnn_layer,
                         args.n_layers,
                         args.layer_dim,
                         input_feature_dim=features.size(1),
                         n_classes=num_labels).to(device)
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
        t_1 = time.time()
        train_pass(gnn_model,
                   optimizer,
                   train_blocks,
                   features,
                   masks['train_indices'],
                   labels,
                   n_train_points,
                   args.construct_mfgs)
        train_time = time.time() - t_1

        (train_loss, train_acc, val_loss, val_acc, test_loss, test_acc) = \
            infer_pass(gnn_model,
                       eval_blocks,
                       features,
                       masks,
                       labels,
                       args.construct_mfgs)

        result_message = (
            f"iteration [{train_iter_idx}/{args.train_iters}] | "
        )
        result_message += ', '.join([
            f"Loss: "
            f"train={train_loss:.4f}, "
            f"valid={val_loss:.4f} "
            f"test={test_loss:.4f} "
            f" | "
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
