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
            lock = kwargs["lock"]
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
            lock.release()
        
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
            lock = kwargs["lock"]
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
            lock.release()
    
    lock = Lock()
    barrier = Barrier(world_size)
    mp_dict = run_workers(gather_grads, world_size, backend=backend, lock=lock, barrier=barrier)
    for rank in range(1, world_size):
        for i in range(len(mp_dict["result_0"])):
            assert torch.all(torch.eq(mp_dict["result_0"][i], mp_dict[f"result_{rank}"][i]))


@pytest.mark.parametrize("backend", ["ccl"])
@pytest.mark.parametrize("world_size", [2, 4, 8])
@sar_test
def test_all_to_all_single_node_partition_data_manager(world_size, backend):
    """
    Checks whether all_to_all operation works as expected. Test is
    designed is such a way, that after calling all_to_all, each worker
    should receive a list of tensors with values equal to their rank
    
    It uses precall and callback functions from PartitionDataManager object
    """
    def all_to_all(mp_dict, rank, world_size, tmp_dir, **kwargs):
        try:
            lock = kwargs["lock"]
            initialize_worker(rank, world_size, tmp_dir, backend=kwargs["backend"],
                            barrier=kwargs["barrier"])
            lock.acquire()
            partition_data_manager = sar.PartitionDataManager(rank, lock)
            send_tensors_list = [torch.tensor([x] * world_size) for x in range(world_size)]
            recv_tensors_list = [torch.tensor([-1] * world_size) for _ in range(world_size)]
            sar.comm.all_to_all(recv_tensors_list, send_tensors_list,
                                precall_func=partition_data_manager.precall_func,
                                callback_func=partition_data_manager.callback_func)
            mp_dict[f"result_{rank}"] = recv_tensors_list
            lock.release()
        except Exception as e: 
            mp_dict["traceback"] = str(traceback.format_exc())
            mp_dict["exception"] = e
            lock.release()
    
    lock = Lock()
    barrier = Barrier(world_size)
    mp_dict = run_workers(all_to_all, world_size, backend=backend, lock=lock, barrier=barrier)
    for rank in range(world_size):
        for tensor in mp_dict[f"result_{rank}"]:
            assert torch.all(torch.eq(tensor, torch.tensor([rank] * world_size)))
            
            
@pytest.mark.parametrize("backend", ["ccl"])
@pytest.mark.parametrize("world_size", [2, 4, 8])
@sar_test
def test_all_to_all_single_node_graph_shard_manager(world_size, backend):
    """
    Checks whether all_to_all operation works as expected. Test is
    designed is such a way, that after calling all_to_all, each worker
    should receive a list of tensors with values equal to their rank
    
    It uses precall and callback functions from GraphShardManager object
    """
    def all_to_all(mp_dict, rank, world_size, tmp_dir, **kwargs):
        try:
            lock = kwargs["lock"]
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
            partition_data_manager.features = sar.PointerTensor(partition_data_manager.features, 
                                                                pointer=fgm.pointer_list, 
                                                                linked=fgm.linked_list)
            
            send_tensors_list = [torch.tensor([x] * world_size) for x in range(world_size)]
            recv_tensors_list = [torch.tensor([-1] * world_size) for _ in range(world_size)]
            sar.comm.all_to_all(recv_tensors_list, send_tensors_list,
                                precall_func=fgm.pause_process,
                                callback_func=fgm.resume_process)
            mp_dict[f"result_{rank}"] = recv_tensors_list
            lock.release()
        except Exception as e: 
            mp_dict["traceback"] = str(traceback.format_exc())
            mp_dict["exception"] = e
            lock.release()
    
    lock = Lock()
    barrier = Barrier(world_size)
    mp_dict = run_workers(all_to_all, world_size, backend=backend, lock=lock, barrier=barrier)
    for rank in range(world_size):
        for tensor in mp_dict[f"result_{rank}"]:
            assert torch.all(torch.eq(tensor, torch.tensor([rank] * world_size)))
            
            
@pytest.mark.parametrize("backend", ["ccl"])
@pytest.mark.parametrize("world_size", [2, 4, 8])
@sar_test
def test_exchange_single_tensor_partition_data_manager(world_size, backend):
    """
    Checks whether exchange_single_tensor operation works as expected.  Test is
    designed in such a way, that after calling exchange_single_tensor, worker
    should receive a tensor with `world_size` elements of its rank value
    
    It uses precall and callback functions from PartitionDataManager object
    """
    def exchange_single_tensor_single_node(mp_dict, rank, world_size, tmp_dir, **kwargs):
        try:
            fail_flag = False
            lock = kwargs["lock"]
            initialize_worker(rank, world_size, tmp_dir, backend=kwargs["backend"])
            lock.acquire()
            partition_data_manager = sar.PartitionDataManager(rank, lock)
            send_idx = rank
            recv_idx = rank
            for _ in range(world_size):
                send_tensor = torch.tensor([send_idx] * world_size)
                recv_tensor = torch.tensor([-1] * world_size)
                sar.comm.exchange_single_tensor(recv_idx, send_idx, recv_tensor, send_tensor,
                                                precall_func=partition_data_manager.precall_func,
                                                callback_func=partition_data_manager.callback_func)
                if torch.all(torch.eq(recv_tensor, torch.tensor([rank] * world_size))).item() is False:
                    fail_flag = True
                    break
                send_idx = (send_idx + 1) % world_size
                recv_idx = (recv_idx - 1) % world_size
            lock.release()
            assert fail_flag == False
        except Exception as e: 
            mp_dict["traceback"] = str(traceback.format_exc())
            mp_dict["exception"] = e
            lock.release()
        
    lock = Lock()
    barrier = Barrier(world_size)
    mp_dict = run_workers(exchange_single_tensor_single_node, world_size, backend=backend, lock=lock, barrier=barrier)


@pytest.mark.parametrize("backend", ["ccl"])
@pytest.mark.parametrize("world_size", [2, 4, 8])
@sar_test
def test_exchange_single_tensor_graph_shard_manager(world_size, backend):
    """
    Checks whether exchange_single_tensor operation works as expected.  Test is
    designed in such a way, that after calling exchange_single_tensor, worker
    should receive a tensor with `world_size` elements of its rank value
    
    It uses precall and callback functions from GraphShardManager object
    """
    def exchange_single_tensor_single_node(mp_dict, rank, world_size, tmp_dir, **kwargs):
        try:
            fail_flag = False
            lock = kwargs["lock"]
            graph_name = 'dummy_graph'
            if rank == 0:
                g = get_random_graph()
                dgl.distributed.partition_graph(g, graph_name, world_size,
                                        tmp_dir, num_hops=1,
                                        balance_edges=True)
            barrier.wait()
            initialize_worker(rank, world_size, tmp_dir, backend=kwargs["backend"])
            lock.acquire()
            partition_data_manager, fgm = load_partition_data_single_node(rank, graph_name, tmp_dir,
                                                                          kwargs["lock"])
            num_labels = partition_data_manager.labels.max() + 1
            sar.comm.all_reduce(num_labels, dist.ReduceOp.MAX, move_to_comm_device=True, 
                        precall_func=partition_data_manager.precall_func, callback_func=partition_data_manager.callback_func)
            partition_data_manager.features = sar.PointerTensor(partition_data_manager.features, 
                                                                pointer=fgm.pointer_list, 
                                                                linked=fgm.linked_list)
            send_idx = rank
            recv_idx = rank
            for _ in range(world_size):
                send_tensor = torch.tensor([send_idx] * world_size)
                recv_tensor = torch.tensor([-1] * world_size)
                sar.comm.exchange_single_tensor(recv_idx, send_idx, recv_tensor, send_tensor,
                                                precall_func=fgm.pause_process,
                                                callback_func=fgm.resume_process)
                if torch.all(torch.eq(recv_tensor, torch.tensor([rank] * world_size))).item() is False:
                    fail_flag = True
                    break
                send_idx = (send_idx + 1) % world_size
                recv_idx = (recv_idx - 1) % world_size
            lock.release()
            assert fail_flag == False
        except Exception as e: 
            mp_dict["traceback"] = str(traceback.format_exc())
            mp_dict["exception"] = e
            lock.release()
        
    lock = Lock()
    barrier = Barrier(world_size)
    mp_dict = run_workers(exchange_single_tensor_single_node, world_size, backend=backend, lock=lock, barrier=barrier)
