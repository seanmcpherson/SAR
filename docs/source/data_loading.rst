.. _data-loading:


Data loading and graph construction
==========================================================
After partitioning the graph using DGL's `partition_graph <https://docs.dgl.ai/en/0.6.x/generated/dgl.distributed.partition.partition_graph.html>`_ function, SAR can load the graph data using :func:`sar.load_dgl_partition_data`. This yields a :class:`sar.common_tuples.PartitionData` object. The ``PartitionData`` object can then be used to construct various types of graph-like objects that can be passed to GNN models. You can construct graph objects to use for distributed full-batch training or graph objects to use for distributed training as follows:

.. contents:: :local:
    :depth: 3


Full-batch training
---------------------------------------------------------------------------------------

Constructing the full graph for sequential aggregation and rematerialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Construct a single distributed graph object of type :class:`sar.core.GraphShardManager`::

    shard_manager = sar.construct_full_graph(partition_data)

..

The ``GraphShardManager`` object encapsulates N DGL graph objects (where N is the number of workers). Each graph object represents the edges incoming from one partition (including the local partition). ``GraphShardManager`` implements the ``update_all`` and ``apply_edges`` methods in addition to several other methods from the standard  ``dgl.heterograph.DGLGraph`` API.  The ``update_all`` and ``apply_edges`` methods implement the sequential aggregation and rematerialization scheme to realize the distributed forward and backward passes. ``GraphShardManager`` can usually be passed to GNN layers instead of ``dgl.heterograph.DGLGraph``. See the :ref:`the distributed graph limitations section<shard-limitations>` for some exceptions.

Constructing Message Flow Graphs (MFGs) for sequential aggregation and rematerialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In node classification tasks, gradients only backpropagate from the labeled nodes. DGL uses the concept of message flow graphs to construct layer-specific bi-partite graphs that update only a subset of nodes in each layer. These are the nodes  that will ultimately affect the output, assuming each node only aggregates messages from its neighbors in every layer.

If training a K-layer GNN on a node classification tasks, you can construct K distributed graph objects that reflect the message flow graphs at each layer using :class:`sar.construct_mfgs`:
::

    class GNNModel(nn.Module):
        def __init__(n_layers: int):
            super().__init__()
            self.convs = nn.ModuleList([
                dgl.nn.SAGEConv(100, 100)
                for _ in range(n_layers)
            ])

        def forward(blocks: List[sar.GraphShardManager], features: torch.Tensor):
            for idx in range(len(self.convs)):
                features = self.convs[idx](blocks[idx], features)
            return features

    K = 3 # number of layers
    gnn_model = GNNModel(K)
    train_blocks = sar.construct_mfgs(partition_data,
                                      global_indices_of_labeled_nodes_in_partition,
                                      K)
    model_out = gnn_model(train_blocks, local_node_features)

..

Using message flow graphs at each layer can substantially lower run-time and memory consumption in node classification tasks with few labeled nodes. 


Constructing full graph or MFGs for one-shot aggregation 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
As described in :ref:`training modes <sar-modes>`, SAR supports doing one-shot distributed aggregation (mode 3). To run in this mode, you should extract the full partition graph from the :class:`sar.core.GraphShardManager` object and use that during training. When using the full graph:
::

    shard_manager = sar.construct_full_graph(partition_data)
    one_shot_graph = shard_manager.get_full_partition_graph()
    del shard_manager
    ## Use one_shot_graph from now on. 

..

When using MFGs:
::

    train_blocks = sar.construct_mfgs(partition_data,
                                      global_indices_of_labeled_nodes_in_partition,
                                      n_layers)
    one_shot_blocks = [block.get_full_partition_graph() for block in train_blocks]
    del train_blocks
    ## Use one_shot_blocks from now on

..


Sampling-based training
---------------------------------------------------------------------------------------

For sampling-based training, use the dataloader provided by SAR: :func:`sar.DataLoader` to construct globally-sampled graphs. The sampled graphs are vanilla DGL graphs that reside solely on the local machines. SAR provides a global neighbor sampler: :class:`sar.DistNeighborSampler` that defines the sampling process from the distributed graph. A typical use case is:

::

   shard_manager = sar.construct_full_graph(partition_data)

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
	    # ...

..


.. _inference-sar-train-distdgl:

Full-graph inference with SAR and mini-batch training with DistDGL
---------------------------------------------------------------------------------------
SAR might also be utilized just for model evaluation. It is preferable to evaluate the model on the entire graph while performing mini-batch distributed training with the DGL package. To accomplish this, SAR can turn a `DistGraph <https://docs.dgl.ai/api/python/dgl.distributed.html#dgl.distributed.DistGraph>`_ object into a GraphShardManager object, allowing for distributed full-graph inference. The procedure is simple since no further steps are required because the model parameters are already synchronized during inference. You can use :func:`sar.convert_dist_graph` in the following way to perform full-graph inference:
::

    class GNNModel(nn.Module):
        def __init__(n_layers: int):
            super().__init__()
            self.convs = nn.ModuleList([
                dgl.nn.SAGEConv(100, 100)
                for _ in range(n_layers)
            ])

        # forward function prepared for mini-batch training
        def forward(blocks: List[DGLBlock], features: torch.Tensor):
            h = features
            for idx, (layer, block) in enumerate(zip(self.convs, blocks)):
                h = self.convs[idx](blocks[idx], h)
            return h
        
        # implement inference function for full-graph input 
        def full_graph_inference(graph: sar.GraphShardManager, featues: torch.Tensor):
            h = features
            for idx, layer in enumerate(self.convs):
                h = layer(graph, h)
            return h

    # model wrapped in pytorch DistributedDataParallel
    gnn_model = th.nn.parallel.DistributedDataParallel(GNNModel(3))
    
    # Convert DistGraph into GraphShardManager
    gsm = sar.convert_dist_graph(g).to(device)

    # Access to model through DistributedDataParallel module field
    model_out = gnn_model.module.full_graph_inference(gsm, local_node_features)

..


.. _single-node-training-graph-construction:

Single-node training
---------------------------------------------------------------------------------------
For single-node training, firstly create object of type :class:`sar.PartitionDataManager` in each worker
::

    partition_data_manager = sar.PartitionDataManager(rank, lock)

..

The ``PartitionDataManager`` object is used to manage saving and loading files from disk. Initialize it with node's rank and lock (``multiprocessing.Lock``) object used to synchronize processes. It takes care of saving and loading from disk following files:

- partition data (created with :func:`sar.load_dgl_partition_data`)
- train, validation, and test masks
- labels
- features and tensors used during training 

You need to explicilty assign above values to the ``PartitionDataManager`` object
::

    # Specify partition_data
    partition_data_manager.partition_data = sar.load_dgl_partition_data(json_file, rank, device)

    # Specify train, validation, and test masks
    partition_data_manager.masks = {}
    for mask_name, indices_name in zip(['train_mask', 'val_mask', 'test_mask'],
                                       ['train_indices', 'val_indices', 'test_indices']):
        boolean_mask = sar.suffix_key_lookup(partition_data_manager.partition_data.node_features,
                                             mask_name)
        partition_data_manager.masks[indices_name] = boolean_mask.nonzero(
            as_tuple=False).view(-1).to(device)

    # Specyfy labels
    partition_data_manager.labels = sar.suffix_key_lookup(
                                        partition_data_manager.partition_data.node_features,
                                        'labels').long().to(device)

    # Specyfy features
    partition_data_manager.features = sar.suffix_key_lookup(
                                        partition_data_manager.partition_data.node_features,
                                        'features').to(device)
    # ...
    # Creating model
    # Synchronizing model's parameters
    # etc.

    full_graph_manager = sar.construct_full_graph(partition_data_manager.partition_data, 
                                                  partition_data_manager=partition_data_manager, 
                                                  lock=lock).to(device)

..

Constructing ``GraphShardManager`` object in single-node training is similar to creating it during full-batch training. Besides ``PartitionData`` object, you also need to pass an object of ``PartitionDataManager`` class, and a lock (``multiprocessing.Lock``) object used to synchronize processes


Such created ``GraphShardManager`` object can be easily used as a drop-in replacement for DGL’s native graph object
::

    logits = gnn_model(full_graph_manager, partition_data_manager.features)

..


Relevant methods
---------------------------------------------------------------------------------------

.. currentmodule:: sar


.. autosummary::
   :toctree: Data loading and graph construction
   :template: distneighborsampler


   load_dgl_partition_data
   construct_full_graph
   construct_mfgs
   convert_dist_graph
   DataLoader
   DistNeighborSampler
   PartitionDataManager
   suffix_key_lookup
