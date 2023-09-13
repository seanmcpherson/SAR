.. _quick-start:

Quick start guide
===============================
Follow the following steps to enable distributed training in your DGL code:

.. contents::
    :depth: 2
    :local:
    :backlinks: top

Partition the graph
----------------------------------
Partition the graph using DGL's `partition_graph <https://docs.dgl.ai/en/0.6.x/generated/dgl.distributed.partition.partition_graph.html>`_ function. See `here <https://github.com/dmlc/dgl/blob/master/examples/pytorch/graphsage/dist/partition_graph.py>`_ for an example. The number of partitions should be the same as the number of training machines/workers that will be used. SAR requires consecutive node indices in each partition, and requires that the partition information include the one-hop neighborhoods of all nodes in the partition. Setting ``num_hops = 1`` and ``reshuffle = True`` (in DGL < 1.0) in the call to ``partition_graph`` takes care of these requirements. ``partition_graph`` yields a directory structure with the partition information and a .json file ``graph_name.json``.


An example of partitioning the ogbn-arxiv graph in two parts: ::
  
    import dgl
    import torch
    from ogb.nodeproppred import DglNodePropPredDataset

    dataset = DglNodePropPredDataset(name='ogbn-arxiv')
    graph = dataset[0][0]
    graph = dgl.to_bidirected(graph, copy_ndata=True)
    graph = dgl.add_self_loop(graph)

    labels = dataset[0][1].view(-1)
    split_idx = dataset.get_idx_split()


    def _idx_to_mask(idx_tensor):
        mask = torch.BoolTensor(graph.number_of_nodes()).fill_(False)
        mask[idx_tensor] = True
        return mask


    train_mask, val_mask, test_mask = map(
        _idx_to_mask, [split_idx['train'], split_idx['valid'], split_idx['test']])
    features = graph.ndata['feat']
    graph.ndata.clear()
    for name, val in zip(['train_mask', 'val_mask', 'test_mask', 'labels', 'features'],
                         [train_mask, val_mask, test_mask, labels, features]):
        graph.ndata[name] = val

    dgl.distributed.partition_graph(
        graph, 'arxiv', 2, './test_partition_data/', num_hops=1) # use reshuffle=True in DGL < 1.0

..

Note that we add the labels, and the train/test/validation masks as node features so that they get split into multiple parts alongside the graph.


Initialize communication
----------------------------------
SAR uses the `torch.distributed <https://pytorch.org/docs/stable/distributed.html>`_ package to handle all communication. See the :ref:`Communication Guide <comm-guide>`  for more information on the communication routines. We require the IP address of the master worker/machine (the machine with rank 0) to initialize the ``torch.distributed`` package. In an environment with a networked file system where all workers/machines share a common file system, we can communicate the master's IP address through the file system. In that case, use :func:`sar.nfs_ip_init` to obtain the master ip address.

Initialize the communication through a call to :func:`sar.initialize_comms` , specifying the current worker index, the total number of workers (which should be the same as the number of partitions from step 1), the master's IP address, and the communication device. The latter is the device on which SAR should place the tensors before sending them through the communication backend.   For example: ::

  if backend_name == 'nccl':
      comm_device = torch.device('cuda')
  else:
      comm_device = torch.device('cpu')
  master_ip_address = sar.nfs_ip_init(rank, path_to_ip_file)
  sar.initialize_comms(rank, world_size, master_ip_address, backend_name, comm_device)
  
.. 

``backend_name`` can be ``ccl``, ``nccl``, ``mpi`` or ``gloo``.

When running single-node training :func:`sar.initialize_comms` must be passed two additional parameters. The first one is ``shared_file``, which specifies the path to the file for inter-process communication. The second one is ``barrier``, which is used for synchronizing processes.
::

  sar.initialize_comms(rank, world_size, master_ip_address, backend_name, comm_device,
                        shared_file=shared_file, barrier=barrier)

..



Load partition data and construct graph
-----------------------------------------------------------------
Use :func:`sar.load_dgl_partition_data` to load one graph partition from DGL's partition data in each worker. :func:`sar.load_dgl_partition_data` returns a :class:`sar.common_tuples.PartitionData` object that contains all the information about the partition.

There are several ways to construct a distributed graph-like object from ``PartitionData``. See :ref:`constructing distributed graphs <data-loading>` for more details. Here we will use the simplest method:  :func:`sar.construct_full_graph` which returns a :class:`sar.core.GraphShardManager` object which implements many of the GNN-related functionality of DGL's native graph objects. ``GraphShardManager`` can thus be used as a drop-in replacement for DGL's native graphs or it can be passed to SAR's samplers and data loaders to construct graph mini-batches.

::
   
    partition_data = sar.load_dgl_partition_data(
        json_file_path, #Path to .json file created by DGL's partition_graph
        rank, #Worker rank
        device #Device to place the partition data (CPU or GPU)
    )
    shard_manager = sar.construct_full_graph(partition_data)
    
.. 

Full-batch training
---------------------------------------------------------------------------
Full-batch training using SAR follows a very similar pattern as single-host training. Instead of using a vanilla DGL graph, we use a :class:`sar.core.GraphShardManager`. After initializing the communication backend, loading graph data and constructing the distributed graph, a simple training loop is  ::

  gnn_model = construct_GNN_model(...)
  optimizer = torch.optim.Adam(gnn_model.parameters(),..)
  sar.sync_params(gnn_model)
  for train_iter in range(n_train_iters):
     model_out = gnn_model(shard_manager,features)
     loss = calculate_loss(model_out,labels)
     optimizer.zero_grad()
     loss.backward()
     sar.gather_grads(gnn_model)
     optimizer.step()

..

In a distributed setting, each worker will construct the GNN model. Before training, we should synchronize the model parameters across all workers. :func:`sar.sync_params` is a convenience function that does just that. At the end of every training iteration, each worker needs to gather and sum the parameter gradients from all other workers before making the parameter update. This can be done using :func:`sar.gather_grads`.

See :ref:`training modes <sar-modes>` for the different full-batch training modes.

Sampling-based or mini-batch training
---------------------------------------------------------------------------
A simple sampling-based training loop looks as follows:
      
::

   neighbor_sampler = sar.DistNeighborSampler(
   [15, 10, 5], #Fanout for every layer
   input_node_features={'features': features}, #Input features to add to srcdata of first layer's sampled block
   output_node_features={'labels': labels} #Output features to add to dstdata of last layer's sampled block
   )

   dataloader = sar.DataLoader(
        shard_manager, #Distributed graph
        train_nodes, #Global indices of nodes that will form the root of the sampled graphs. In node classification, these are the labeled nodes
        neighbor_sampler, #Distributed sampler
        batch_size)

   for blocks in dataloader:
     output = gnn_model(blocks)
     loss = calculate_loss(output,labels)
     optimizer.zero_grad()
     loss.backward()
     sar.gather_grads(gnn_model)
     optimizer.step()

..		


We use :class:`sar.DistNeighborSampler` to construct a distributed sampler and :func:`sar.DataLoader` to construct an iterator that retrurn standard local DGL blocks constructed from the distributed graph.  


Single-node training
---------------------------------------------------------------------------
Single-node training enables GNNs training on a very larg graphs on a single machine. This is achieved by running each worker as a separate process. At one time only one worker/process can store its data in a main memory (other processes keep their data saved on disk). SAR uses locks and barriers for synchronizing processes (only one process can aquire lock at a given time). When a process triggers a collective communication call, it sends the data to other processes using torch.distributed file based communication, saves its data to the disk and releases a lock. Then, the next process can load its part of the data from disk and continue its work.

In order to construct ``GraphShardManager`` object during single-node training user is additionally required to prepare ``PartitionDataManager`` object. This object is needed to manage saving and loading data from disk by each process. You can read the :ref:`constructing distributed graphs for single-node training <single-node-training-graph-construction>` section for a detailed explenation of how to construct ``PartitionDataManager`` and ``GraphShardManager`` objects.
Assuming you created ``GraphShardManager`` and ``PartitionDataManager`` objects, a simple training loop might look as follows:

::

   for epoch in range(num_epochs):
        partition_data_manager.features = sar.PointerTensor(partition_data_manager.features, 
                                                            pointer=full_graph_manager.pointer_list, 
                                                            linked=full_graph_manager.linked_list)

        logits = gnn_model(full_graph_manager, partition_data_manager.features)
        loss = F.cross_entropy(logits[partition_data_manager.masks['train_indices']],
                               partition_data_manager.labels[partition_data_manager.masks['train_indices']], reduction='sum')/n_train_points

        optimizer.zero_grad()
        loss.backward()
        sar.gather_grads(gnn_model, 
                         precall_func=full_graph_manager.pause_process,
                         callback_func=full_graph_manager.resume_process)
        optimizer.step()

        logits = torch.Tensor(logits)
        partition_data_manager.features = torch.Tensor(partition_data_manager.features)
        full_graph_manager.pointer_list = []
        full_graph_manager.linked_list = []

        partition_data_manager.remove_files()

..	

SAR uses class named :class:`sar.PointerTensor`, which inherits from ``torch.Tensor`` in order to keep track of features and all of the tensors calculated during an epoch (mechanism needed to properly save every tensor on the disk). ``GraphShardManager`` stores those tensors in two lists called ``pointer_list`` and ``linked_list``. Both lists must be cleand at the end of the epoch.
You should use :func:`remove_files` function of ``PartitionDataManager`` class to clear the disk from all of the saved files, at the end of the epoch.
During single-node training every function performing collective communications must be passed ``precall_func`` and ``callback_func``, which are responsible for saving and loading data from disk.

To learn more, read the :ref:`single-node training <single-node-training>` section.


For complete examples, check the `examples folder <https://github.com/IntelLabs/SAR/tree/main/examples>`_ in the Git repository.
