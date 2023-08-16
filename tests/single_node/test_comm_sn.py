from multiprocessing import Lock, Barrier
from multiprocessing_utils import *
from copy import deepcopy
import torch
import torch.nn.functional as F
import torch.distributed as dist
import os
import sar
import dgl
from models import GNNModel
from base_utils import *


@pytest.mark.parametrize("backend", ["ccl", "gloo"])
@pytest.mark.parametrize("world_size", [2, 4, 8])
@sar_test
def test_sync_params_single_node(backend, world_size):
    """
    Checks whether model's parameters are the same across all
    workers after calling sync_params function. Parameters of worker 0
    should be copied to all workers, so its parameters before and after
    sync_params should be the same
    """
    def sync_params(mp_dict, rank, world_size, tmp_dir, **kwargs):
        try:
            initialize_worker(rank, world_size, tmp_dir, backend=kwargs["backend"],
                            barrier=kwargs["barrier"])
            lock.acquire()
            model = GNNModel(16, 4)
            partition_data_manager = sar.PartitionDataManager(rank, lock)
            if rank == 0:   
                mp_dict[f"result_{rank}"] = deepcopy(model.state_dict())
            sar.sync_params(model, precall_func=partition_data_manager.precall_func,
                            callback_func=partition_data_manager.callback_func)
            if rank != 0:
                mp_dict[f"result_{rank}"] = model.state_dict()
            lock.release()
        except Exception as e:
            mp_dict["traceback"] = str(traceback.format_exc())
            mp_dict["exception"] = e
        
    lock = Lock()
    barrier = Barrier(world_size)
    mp_dict = run_workers(sync_params, world_size, backend=backend, lock=lock, barrier=barrier)
    for rank in range(1, world_size):
        for key in mp_dict[f"result_0"].keys():
            assert torch.all(torch.eq(mp_dict[f"result_0"][key], mp_dict[f"result_{rank}"][key]))


@pytest.mark.parametrize("backend", ["ccl"])
@pytest.mark.parametrize("world_size", [2, 4, 8])
@sar_test
def test_gather_grads_single_node(world_size, backend):
    """
    Checks whether parameter's gradients are the same across all
    workers after calling gather_grads function 
    """
    def gather_grads(mp_dict, rank, world_size, tmp_dir, **kwargs):
        try:
            graph_name = 'dummy_graph'
            if rank == 0:
                g = get_random_graph()
                dgl.distributed.partition_graph(g, graph_name, world_size,
                                        tmp_dir, num_hops=1,
                                        balance_edges=True)
            barrier.wait()
            initialize_worker(rank, world_size, tmp_dir, backend=kwargs["backend"],
                            barrier=kwargs["barrier"])
            lock.acquire()
            partition_data_manager, fgm = load_partition_data_single_node(rank, graph_name, tmp_dir,
                                                                          kwargs["lock"])
            num_labels = partition_data_manager.labels.max() + 1
            sar.comm.all_reduce(num_labels, dist.ReduceOp.MAX, move_to_comm_device=True, 
                        precall_func=partition_data_manager.precall_func, callback_func=partition_data_manager.callback_func)
            
            model = GNNModel(partition_data_manager.features.shape[1], num_labels)
            sar.sync_params(model, precall_func=partition_data_manager.precall_func,
                            callback_func=partition_data_manager.callback_func)
            
            
            partition_data_manager.features = sar.PointerTensor(partition_data_manager.features, 
                                                                pointer=fgm.pointer_list, 
                                                                linked=fgm.linked_list)
            sar_logits = model(fgm, partition_data_manager.features)
            sar_loss = F.cross_entropy(sar_logits, partition_data_manager.labels)
            sar_loss.backward()
            sar.gather_grads(model, precall_func=fgm.pause_process,
                             callback_func=fgm.resume_process)
            mp_dict[f"result_{rank}"] = [torch.tensor(x.grad) for x in model.parameters()]
            lock.release()
        except Exception as e:
            mp_dict["traceback"] = str(traceback.format_exc())
            mp_dict["exception"] = e
    
    lock = Lock()
    barrier = Barrier(world_size)
    mp_dict = run_workers(gather_grads, world_size, backend=backend, lock=lock, barrier=barrier)
    for rank in range(1, world_size):
        for i in range(len(mp_dict["result_0"])):
            assert torch.all(torch.eq(mp_dict["result_0"][i], mp_dict[f"result_{rank}"][i]))
