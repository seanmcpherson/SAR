import torch

INPLACE_FUNCTIONS = [
    torch.Tensor.resize_,
    torch.Tensor.copy_,
    torch.Tensor.storage, 
    torch.Tensor.detach, 
    torch.Tensor.set_, 
    torch.Tensor.requires_grad, 
    torch.Tensor.data_ptr, 
    torch.Tensor.new 
]

class PointerTensor(torch.Tensor):
    def __init__(self, data, pointer=[], linked = [], base_tensor = None, func="", **kwargs):
        self._pointer = pointer
        self._linked = linked
        if base_tensor is not None:
            self._linked.append((base_tensor, self))
        else:
            self._pointer.append(self)
        self._func = func

    @staticmethod
    def __new__(cls, x, pointer=[], linked = [], base_tensor = None,  *args, **kwargs):
        return super().__new__(cls, x, *args, **kwargs)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        pointers = tuple(a._pointer for a in args if hasattr(a, '_pointer'))
        if len(pointers) == 0:
            pointers = [[]]
        links = tuple(a._linked for a in args if hasattr(a, '_linked'))
        if len(links) == 0:
            links = [[]]
        parent = super().__torch_function__(func, types, args, kwargs)
        if not type(parent) in [torch.Tensor, cls]:
            return parent

        base_tensor = None
        for pointer in pointers[0]:
            if hasattr(parent, 'data_ptr'):
                if parent.storage().data_ptr() == pointer.storage().data_ptr():
                    base_tensor = pointer
                    break
        
        if func not in INPLACE_FUNCTIONS and not hasattr(parent, '_pointer'):
            parent.__init__([], pointer=pointers[0], linked = links[0], 
                            base_tensor= base_tensor, func=func)
        
        return parent
