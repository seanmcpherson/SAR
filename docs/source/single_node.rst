.. _single-node-training:


Single-node training
=======================
Single-node training enables GNNs training on a very larg graphs on a single machine. This is achieved by running each worker as a separate process. At one time only one worker/process can store its data in a main memory (other processes keep their data saved on disk). SAR uses lock and a barrier for synchronizing processes. That is why you need to create them at the beginning of the program.

::

    from multiprocessing import Process, Lock, Barrier

    lock = Lock()
    barrier = Barrier(args.world_size)

    for rank_idx in range(args.world_size):
        p = Process(target=run, args=(args,rank_idx, lock, barrier))
        p.start()

..
    
where the ``run`` function is an entry point to the script.

In a single-node training, you also need to make a call to a  :func:`sar.initialize_comms` function, in order to initialize the torch.distributed process group. However, you need to pass two additional arguments. One of them is ``shared_file``, which specifies the path to the file for inter-process communication used by torch.districuted package. The second one is ``barrier``, which is needed to synchronize processes.

::

    master_ip_address = sar.nfs_ip_init(rank, args.ip_file)
    sar.initialize_comms(rank, world_size, master_ip_address, backend_name, comm_device,
                            shared_file=shared_file, barrier=barrier)

..

After initializing torch.distributed process group, each worker/process should acquire a ``lock``, because from this point only one process should be working at one time:

::

    lock.acquire()

..

After acquiring the lock for the first time, process should start preparing data for a training. SAR uses :class:`sar.PartitionDataManager` class to manage loading and saving data to disk. It is responsible for saving and loading following files:

- partition data (created with :func:`sar.load_dgl_partition_data`)
- train, validation, and test masks
- labels
- features and tensors used during training

You need to explicilty assign above values to the ``PartitionDataManager``. Moreover, you need to create ``PartitionDataManager`` to properly construct ``GraphShardManager`` object. Read the :ref:`constructing distributed graphs for single-node training <single-node-training-graph-construction>` section for a detailed explenation of how to construct ``PartitionDataManager`` and ``GraphShardManager`` objects.


When a process triggers a collective communication call, it sends the data to other processes using torch.distributed file based communication, saves its data to the disk and releases a lock. Then, the next process can load its part of the data from disk and continue its work. That is why during single-node training every function performing collective communications (or point-to-point communication in case of a ``gloo`` backend) must be passed a ``precall_func`` and ``callback_func`` parameters. Former is a function responsible for saving data to disk by a process after sending data to another processes. The latter is a function used to load data back from disk, when the communication routine is finished. Before training is started and ``GraphShardManager`` object is created, you should use functions :func:`precall_func` and :func:`callback_func` functions which are defined in ``PartitionDataManager`` class. For instance, when synchronizing model's parameters

::

    partition_data_manager = sar.PartitionDataManager(rank, lock)
    # ...
    sar.sync_params(gnn_model, precall_func=partition_data_manager.precall_func, callback_func=partition_data_manager.callback_func)

..

After creating ``GraphShardManager`` object i.e. during training you must use :func:`pause_process` and :func:`resume_process` functions, which are defined in ``GraphShardManager`` class. For instace:

::

    sar.gather_grads(gnn_model,
                     precall_func=full_graph_manager.pause_process,
                     callback_func=full_graph_manager.resume_process)

..


Simple training loop might look as follows:

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

SAR uses class named :class:`sar.PointerTensor`, which inherits from ``torch.Tensor`` in order to keep track of features and all of the tensors calculated during an epoch (mechanism needed to properly save every tensor on the disk). ``GraphShardManager`` stores those tensors in two lists called ``pointer_list`` and ``linked_list``. Both lists must be cleand at the end of each epoch.
You should use :func:`remove_files` function of ``PartitionDataManager`` class to clear the disk from all of the saved files, at the end of each epoch.
During single-node training every function performing collective communications must be passed ``precall_func`` and ``callback_func``, which are responsible for saving and loading data from disk. As you can see, since the ``GraphShardManager`` object is already created, script is using its :func:`pause_process` and :func:`resume_process` functions as precall_func and callback_func.


The last important thing to do, is to remember to make each process release a lock at the very end of the program. This is necessary for other processes waiting for the lock.

::

    lock.release()
    
..

Relevant classes and methods
---------------------------------------------------------------------------


.. autosummary::
   :toctree: Single Node Training
   :template: graphshardmanager
	     
   sar.GraphShardManager
   sar.PartitionDataManager