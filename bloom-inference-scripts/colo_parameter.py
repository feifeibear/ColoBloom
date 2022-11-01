import torch

from typing import Optional, overload, TypeVar, Union
import bitsandbytes as bnb

from colossalai.tensor.colo_tensor import ColoTensor
from colossalai.tensor.const import TensorType
from colossalai.tensor import ColoTensorSpec
from colossalai.tensor.param_op_hook import ParamOpHookManager
from torch import Tensor, device, dtype, nn

def filter_args(func, *args):
    return [arg for arg in args if func(arg)]


def replace_args(args, kwargs, new_args):
    args = new_args[:len(args)]
    for k, v in zip(kwargs.keys(), new_args[len(args):]):
        kwargs[k] = v
    return tuple(args), kwargs

T = TypeVar("T", bound="torch.nn.Module")


class ColoParameter(ColoTensor, torch.nn.Parameter):
    r"""A kind of ColoTensor to be considered as a module parameter.

    """

    def __new__(cls,
                data: Optional[torch.Tensor] = None,
                requires_grad: bool = True,
                spec: ColoTensorSpec = None) -> 'ColoParameter':
        if data is None:
            data = torch.empty(0)
        return torch.Tensor._make_subclass(cls, data, requires_grad)

    def __init__(self,
                 data: Optional[torch.Tensor] = None,
                 requires_grad: bool = True,
                 spec: ColoTensorSpec = None) -> None:
        ColoTensor.__init__(self, data, spec)
        self._type = TensorType.MODEL
        # a list contains modules sharing this ColoParameter with others.
        self._shared_param_modules = []

    @property
    def shared_param_modules(self):
        return self._shared_param_modules

    @staticmethod
    def from_torch_tensor(tensor: torch.Tensor,
                          requires_grad: bool = True,
                          spec: ColoTensorSpec = None) -> 'ColoParameter':
        tensor = tensor.as_subclass(ColoParameter)
        tensor.__init__(tensor, requires_grad=requires_grad, spec=spec)
        return tensor

    def __repr__(self):
        return f'ColoParameter: {ColoTensor.__repr__(self)}'

    @classmethod
    def __torch_function__(cls, func, types, args=..., kwargs=None):
        if ParamOpHookManager.has_hook():
            if not func.__name__.startswith('__'):
                if kwargs is None:
                    kwargs = {}
                params = filter_args(lambda arg: isinstance(arg, ColoParameter), *args, *kwargs.values())
                if len(params) > 0:
                    with torch._C.DisableTorchFunction():
                        new_args = ParamOpHookManager.pre_op(params, *args, *kwargs.values())
                    args, kwargs = replace_args(args, kwargs, new_args)
                    ret = super().__torch_function__(func, types, args, kwargs)
                    with torch._C.DisableTorchFunction():
                        ret = ParamOpHookManager.post_op(params, ret)
                    return ret
        return super().__torch_function__(func, types, args, kwargs)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            with torch._C.DisableTorchFunction():
                data = self.data.clone()
            tensor = ColoParameter(data,
                                   self.requires_grad,
                                   spec=ColoTensorSpec(self.get_process_group(), self.dist_spec, self.compute_spec))
            memo[id(self)] = tensor
            return tensor

    def __reduce_ex__(self, proto):
        # Adapted from torch._utils._rebuild_parameter
        # def _rebuild_colo_parameter(data, requires_grad, backward_hooks):
        #     colo_param = ColoParameter(data, requires_grad)
        #     colo_param._backward_hooks = backward_hooks
        #     return colo_param

        # return (
        #     _rebuild_colo_parameter,
        #     (self.data, self.requires_grad, OrderedDict())
        # )

        # TODO(jzy) we don't support object reflection now.
        # distspec cannot be pickled or rebuilt because it's tightly connected to runtime attribute `process_group`.
        raise NotImplementedError



class Int8Params(torch.nn.Parameter):
    def __new__(
        cls,
        data=None,
        requires_grad=True,
        has_fp16_weights=False,
        CB=None,
        SCB=None,
    ):
        cls.has_fp16_weights = has_fp16_weights
        cls.CB = None
        cls.SCB = None
        if data is None:
            data = torch.empty(0)
        return torch.Tensor._make_subclass(cls, data, requires_grad)

    def __init__(
        cls,
        data=None,
        requires_grad=True,
        has_fp16_weights=False,
        CB=None,
        SCB=None,
    ):
        super().__init__()
        cls.has_fp16_weights = has_fp16_weights
        cls.CB = None
        cls.SCB = None
        
        
    def cuda(self, device):
        if self.has_fp16_weights:
            return super().cuda(device)
        else:
            # we store the 8-bit rows-major weight
            # we convert this weight to the turning/ampere weight during the first inference pass
            B = self.data.contiguous().half().cuda(device)
            CB, CBt, SCB, SCBt, coo_tensorB = bnb.functional.double_quant(B)
            del CBt
            del SCBt
            self.data = CB
            setattr(self, "CB", CB)
            setattr(self, "SCB", SCB)

        return self

    @overload
    def to(
        self: T,
        device: Optional[Union[int, device]] = ...,
        dtype: Optional[Union[dtype, str]] = ...,
        non_blocking: bool = ...,
    ) -> T:
        ...

    @overload
    def to(self: T, dtype: Union[dtype, str], non_blocking: bool = ...) -> T:
        ...

    @overload
    def to(self: T, tensor: Tensor, non_blocking: bool = ...) -> T:
        ...

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(
            *args, **kwargs
        )
        
        if (
            device is not None
            and device.type == "cuda"
            and self.data.device.type == "cpu"
        ):
            return self.to(device)
        else:
            new_param = Int8Params(
                super().to(
                    device=device, dtype=dtype, non_blocking=non_blocking
                ),
                requires_grad=self.requires_grad,
                has_fp16_weights=self.has_fp16_weights,
            )
            
            new_param.CB = self.CB
            new_param.SCB = self.SCB
            return new_param

class ColoInt8Parameter(ColoTensor, Int8Params):
    r"""A kind of ColoTensor to be considered as a module parameter.

    """

    def __new__(cls,
                data: Optional[torch.Tensor] = None,
                requires_grad: bool = True,
                spec: ColoTensorSpec = None,
                ) -> 'ColoInt8Parameter':
        if data is None:
            data = torch.empty(0)
        return torch.Tensor._make_subclass(cls, data, requires_grad)

    def __init__(self,
                 data: Optional[torch.Tensor] = None,
                 requires_grad: bool = True,
                 spec: ColoTensorSpec = None,
                 ) -> None:
        ColoTensor.__init__(self, data, spec)
        Int8Params.__init__(self, device)
        self._type = TensorType.MODEL
        # a list contains modules sharing this ColoInt8Parameter with others.
        self._shared_param_modules = []

    @property
    def shared_param_modules(self):
        return self._shared_param_modules

    @staticmethod
    def from_torch_tensor(tensor: torch.Tensor,
                          requires_grad: bool = True,
                          spec: ColoTensorSpec = None) -> 'ColoInt8Parameter':
        tensor = tensor.as_subclass(ColoInt8Parameter)
        tensor.__init__(tensor, requires_grad=requires_grad, spec=spec)
        return tensor

    def __repr__(self):
        return f'ColoInt8Parameter: {ColoTensor.__repr__(self)}'

    @classmethod
    def __torch_function__(cls, func, types, args=..., kwargs=None):
        if ParamOpHookManager.has_hook():
            if not func.__name__.startswith('__'):
                if kwargs is None:
                    kwargs = {}
                params = filter_args(lambda arg: isinstance(arg, ColoInt8Parameter), *args, *kwargs.values())
                if len(params) > 0:
                    with torch._C.DisableTorchFunction():
                        new_args = ParamOpHookManager.pre_op(params, *args, *kwargs.values())
                    args, kwargs = replace_args(args, kwargs, new_args)
                    ret = super().__torch_function__(func, types, args, kwargs)
                    with torch._C.DisableTorchFunction():
                        ret = ParamOpHookManager.post_op(params, ret)
                    return ret
        return super().__torch_function__(func, types, args, kwargs)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            with torch._C.DisableTorchFunction():
                data = self.data.clone()
            tensor = ColoInt8Parameter(data,
                                   self.requires_grad,
                                   spec=ColoTensorSpec(self.get_process_group(), self.dist_spec, self.compute_spec))
            memo[id(self)] = tensor
            return tensor

    def __reduce_ex__(self, proto):
        # Adapted from torch._utils._rebuild_parameter
        # def _rebuild_colo_parameter(data, requires_grad, backward_hooks):
        #     colo_param = ColoInt8Parameter(data, requires_grad)
        #     colo_param._backward_hooks = backward_hooks
        #     return colo_param

        # return (
        #     _rebuild_colo_parameter,
        #     (self.data, self.requires_grad, OrderedDict())
        # )

        # TODO(jzy) we don't support object reflection now.
        # distspec cannot be pickled or rebuilt because it's tightly connected to runtime attribute `process_group`.
        raise NotImplementedError
