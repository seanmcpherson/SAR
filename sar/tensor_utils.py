import torch

INPLACE_FUNCTIONS = [
    torch.Tensor.resize_,
    torch.Tensor.copy_,
    torch.Tensor.storage, 
    torch.Tensor.detach, 
    torch.Tensor.set_, 
    torch.Tensor.requires_grad 
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
        #if func is torch.Tensor.__repr__: # or func is torch.Tensor.__format__:
        #    args = [a.tensor() if hasattr(a, 'tensor') else a for a in args]
        #    return func(*args, **kwargs)
        pointers = tuple(a._pointer for a in args if hasattr(a, '_pointer'))
        if len(pointers) == 0:
            pointers = [[]]
        #import ipdb; ipdb.set_trace()
        parent = super().__torch_function__(func, types, args, kwargs)
        if func not in INPLACE_FUNCTIONS and not hasattr(parent, '_pointer'):
            parent.__init__([], pointer=pointers[0], func=func)
        return parent
