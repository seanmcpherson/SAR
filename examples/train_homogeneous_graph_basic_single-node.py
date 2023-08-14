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

from memory_profiler import profile

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

parser.add_argument('--shared-file', default='./shared_file', type=str,
                    help='Path to a file required by torch.dist for inter-process communication')

parser.add_argument('--backend', default='nccl', type=str, choices=['ccl', 'nccl', 'mpi', 'gloo'],
                    help='Communication backend to use '
                    )

parser.add_argument(
    "--cpu-run", action="store_true",
    help="Run on CPUs if set, otherwise run on GPUs "
)

parser.add_argument('--log-level', default='INFO', type=str,
                    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                    help='SAR log level ')

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

class PartitionDataManager:
    def __init__(self, idx, folder_name_prefix="partition_rank_"):
        self.idx = idx
        self.folder_name = f"{folder_name_prefix}{idx}"
        self._features = None
        self._masks = None
        self._labels = None
        self._partition_data = None

    @property
    def partition_data(self):
        if self._partition_data is None:
            raise ValueError("partition_data not set")
        return self._partition_data
    
    @partition_data.setter
    def partition_data(self, data):
        self._partition_data = data

    @partition_data.deleter
    def partition_data(self):
        del self._partition_data

    @property
    def features(self):
        #if self._features is None:
        #    raise ValueError("features not set")
        return self._features
    
    @features.setter
    def features(self, feats):
        self._features = feats

    @features.deleter
    def features(self):
        del self._features
        self._features = None

    @property
    def labels(self):
        if self._labels is None:
            raise ValueError("labels not set")
        return self._labels
    
    @labels.setter
    def labels(self, labels):
        self._labels = labels
    
    @property
    def masks(self):
        if self._masks is None:
            raise ValueError("masks not set")
        return self._masks
    
    @masks.setter
    def masks(self, masks):
        self._masks = masks
    
    def save(self):
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)
        #if self._features is not None:
        #    torch.save(self._features, os.path.join(self.folder_name, "features.pt"))
        if self._masks is not None:
            torch.save(self._masks, os.path.join(self.folder_name, "masks.pt"))
        if self._labels is not None:
            torch.save(self._labels, os.path.join(self.folder_name, "labels.pt"))
        if self._partition_data is not None:
            torch.save(self._partition_data, os.path.join(self.folder_name, "partition_data.pt"))

    def save_tensor(self, tensor, tensor_name):
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)
        if tensor is not None:
            torch.save(tensor, os.path.join(self.folder_name, tensor_name + ".pt"))

    def delete(self):
        #del self._features
        del self._masks
        del self._labels
        del self._partition_data

    def load(self):
        if not os.path.exists(self.folder_name):
            raise FileNotFoundError("No partition data saved")
        #if os.path.exists(os.path.join(self.folder_name, "features.pt")):
        #    self._features = torch.load(os.path.join(self.folder_name, "features.pt"))
        #else:
        #    print("features not loaded, no file saved")
        if os.path.exists(os.path.join(self.folder_name, "masks.pt")):
            self._masks = torch.load(os.path.join(self.folder_name, "masks.pt"))
        else:
            print("masks not loaded, no file saved")
        if os.path.exists(os.path.join(self.folder_name, "labels.pt")):
            self._labels = torch.load(os.path.join(self.folder_name, "labels.pt"))
        else:
            print("labels not loaded, no file saved")
        if os.path.exists(os.path.join(self.folder_name, "partition_data.pt")):
            self._partition_data = torch.load(os.path.join(self.folder_name, "partition_data.pt"))
        else:
            print("partition_data not loaded, no file saved")

    def load_tensor(self, tensor_name):
        if not os.path.exists(self.folder_name):
            raise FileNotFoundError("No partition data saved")
        if os.path.exists(os.path.join(self.folder_name, tensor_name + ".pt")):
            return torch.load(os.path.join(self.folder_name, tensor_name + ".pt"))
        else: 
            return None


def main():
    args = parser.parse_args()
    
    from multiprocessing import Process, Lock, Barrier

    lock = Lock()
    barrier = Barrier(args.world_size)

    for rank_idx in range(args.world_size):
        p = Process(target=run, args=(args,rank_idx, lock, barrier))
        p.start()

def run(args, rank, lock, barrier):
    print('args', args)
    print('rank', rank)

    use_gpu = torch.cuda.is_available() and not args.cpu_run
    device = torch.device('cuda' if use_gpu else 'cpu')

    sar.logging_setup(logging.getLevelName(args.log_level),
                      rank, args.world_size)
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.NullHandler())
    logger.setLevel(args.log_level)

    # Obtain the ip address of the master through the network file system
    master_ip_address = sar.nfs_ip_init(rank, args.ip_file)
    sar.initialize_comms(rank,
                         args.world_size, master_ip_address,
                         args.backend, shared_file=args.shared_file, barrier=barrier)

    lock.acquire()
    print("Node {} Lock Acquired".format(rank))

    sar.start_comm_thread()
    # Load DGL partition data
    partition_data_manager = PartitionDataManager(rank)
    partition_data_manager.partition_data = sar.load_dgl_partition_data(
        args.partitioning_json_file, rank, device)

    
    # Obtain train,validation, and test masks
    # These are stored as node features. Partitioning may prepend
    # the node type to the mask names. So we use the convenience function
    # suffix_key_lookup to look up the mask name while ignoring the
    # arbitrary node type
    partition_data_manager.masks = {}
    for mask_name, indices_name in zip(['train_mask', 'val_mask', 'test_mask'],
                                       ['train_indices', 'val_indices', 'test_indices']):
        boolean_mask = sar.suffix_key_lookup(partition_data_manager.partition_data.node_features,
                                             mask_name)
        partition_data_manager.masks[indices_name] = boolean_mask.nonzero(
            as_tuple=False).view(-1).to(device)

    partition_data_manager.labels = sar.suffix_key_lookup(partition_data_manager.partition_data.node_features,
                                   'labels').long().to(device)

    # Obtain the number of classes by finding the max label across all workers
    num_labels = partition_data_manager.labels.max() + 1
    
    def precall_func():
        partition_data_manager.save()
        if isinstance(partition_data_manager.features, torch.Tensor): 
            partition_data_manager.save_tensor(partition_data_manager.features, 'features')
            del partition_data_manager.features
        partition_data_manager.delete()
        lock.release()
        print("Node {} Lock Released".format(rank))

    def callback_func(handle):    
        handle.wait()
        lock.acquire()
        print("Node {} Lock Acquired".format(rank))
        partition_data_manager.load()
        features = partition_data_manager.load_tensor('features')
        if features is not None:
            partition_data_manager.features = features

    
    sar.comm.all_reduce(num_labels, dist.ReduceOp.MAX, move_to_comm_device=True, 
                        precall_func=precall_func, callback_func=callback_func)
    print("Node {} Num Labels {}".format(rank, num_labels))

    num_labels = num_labels.item() 
    
    partition_data_manager.features = sar.suffix_key_lookup(partition_data_manager.partition_data.node_features, 'features').to(device)
    
    
    gnn_model = GNNModel(partition_data_manager.features.size(1),
                         args.hidden_layer_dim,
                         num_labels).to(device)
    print('model', gnn_model)

    # Synchronize the model parmeters across all workers
    sar.sync_params(gnn_model, precall_func=precall_func, callback_func=callback_func)

    # Obtain the number of labeled nodes in the training
    # This will be needed to properly obtain a cross entropy loss
    # normalized by the number of training examples
    n_train_points = torch.LongTensor([partition_data_manager.masks['train_indices'].numel()])
    sar.comm.all_reduce(n_train_points, op=dist.ReduceOp.SUM, move_to_comm_device=True, 
                        precall_func=precall_func, callback_func=callback_func)
    n_train_points = n_train_points.item()

    full_graph_manager = sar.construct_full_graph(partition_data_manager.partition_data, 
                                                  partition_data_manager=partition_data_manager, lock=lock).to(device)
    
    #We do not need the partition data anymore
    del partition_data_manager.partition_data
    partition_data_manager.partition_data = None
    
    #full_graph_manager.partition_data_manager = partition_data_manager

    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=args.lr)
    for train_iter_idx in range(args.train_iters):
        logger.debug(f'{rank} : starting training iteration {train_iter_idx}')
        partition_data_manager.features = sar.PointerTensor(partition_data_manager.features, 
                                                            pointer=full_graph_manager.pointer_list, 
                                                            linked=full_graph_manager.linked_list)
        # Train
        t_1 = time.time()
        logits = gnn_model(full_graph_manager, partition_data_manager.features)
        logger.debug(f'{rank} : training iteration complete {train_iter_idx}')
        loss = F.cross_entropy(logits[partition_data_manager.masks['train_indices']],
                               partition_data_manager.labels[partition_data_manager.masks['train_indices']], reduction='sum')/n_train_points

        optimizer.zero_grad()
        loss.backward()
        # Do not forget to gather the parameter gradients from all workers
        sar.gather_grads(gnn_model, 
                         precall_func=full_graph_manager.pause_process, callback_func=full_graph_manager.resume_process)
        optimizer.step()

        logits = torch.Tensor(logits)
        partition_data_manager.features = torch.Tensor(partition_data_manager.features)
        full_graph_manager.pointer_list = []
        full_graph_manager.linked_list = []

        train_time = time.time() - t_1

        # Calculate accuracy for train/validation/test
        results = []
        for indices_name in ['train_indices', 'val_indices', 'test_indices']:
            n_correct = (logits[partition_data_manager.masks[indices_name]].argmax(1) ==
                         partition_data_manager.labels[partition_data_manager.masks[indices_name]]).float().sum()
            results.extend([n_correct, partition_data_manager.masks[indices_name].numel()])

        acc_vec = torch.FloatTensor(results)
        # Sum the n_correct, and number of mask elements across all workers
        sar.comm.all_reduce(acc_vec, op=dist.ReduceOp.SUM, move_to_comm_device=True, 
                            precall_func=full_graph_manager.pause_process, callback_func=full_graph_manager.resume_process)
        (train_acc, val_acc, test_acc) =  \
            (acc_vec[0] / acc_vec[1],
             acc_vec[2] / acc_vec[3],
             acc_vec[4] / acc_vec[5])

        result_message = (
            f"iteration [{train_iter_idx + 1}/{args.train_iters}] | "
        )
        result_message += ', '.join([
            f"train loss={loss}, "
            f"Accuracy: "
            f"train={train_acc} "
            f"valid={val_acc} "
            f"test={test_acc} "
            f" | train time = {train_time} "
            f" |"
        ])
        print(result_message, flush=True)
        
        full_graph_manager.print_metrics()
        
    lock.release()

if __name__ == '__main__':
    main()
