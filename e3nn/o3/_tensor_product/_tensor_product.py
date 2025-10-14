import warnings
from math import sqrt
from typing import List, Optional, Union, Any

import e3nn
import torch
import torch.fx
from e3nn import o3
from e3nn.util import prod
from e3nn.util.codegen import CodeGenMixin
from e3nn.util.jit import compile_mode
from torch import fx
from ._codegen import codegen_tensor_product_left_right, codegen_tensor_product_right
from ._instruction import Instruction


@compile_mode('script')
class TensorProduct(CodeGenMixin, torch.nn.Module):
    r"""Tensor product with parametrized paths and LoRA support.

    Parameters
    ----------
    irreps_in1 : `e3nn.o3.Irreps`
        Irreps for the first input.

    irreps_in2 : `e3nn.o3.Irreps`
        Irreps for the second input.

    irreps_out : `e3nn.o3.Irreps`
        Irreps for the output.

    instructions : list of tuple
        List of instructions ``(i_1, i_2, i_out, mode, train[, path_weight])``.

    in1_var : list of float, Tensor, or None
        Variance for each irrep in ``irreps_in1``. If ``None``, all default to ``1.0``.

    in2_var : list of float, Tensor, or None
        Variance for each irrep in ``irreps_in2``. If ``None``, all default to ``1.0``.

    out_var : list of float, Tensor, or None
        Variance for each irrep in ``irreps_out``. If ``None``, all default to ``1.0``.

    irrep_normalization : {'component', 'norm'}
        The assumed normalization of the input and output representations.

    path_normalization : {'element', 'path'}
        Normalization strategy for paths.

    internal_weights : bool
        whether the `e3nn.o3.TensorProduct` contains its learnable weights as a parameter

    shared_weights : bool
        whether the learnable weights are shared among the input's extra dimensions

    compile_left_right : bool
        whether to compile the forward function, true by default

    compile_right : bool
        whether to compile the ``.right`` function, false by default
    """
    instructions: List[Any]
    shared_weights: bool
    internal_weights: bool
    weight_numel: int
    _specialized_code: bool
    _optimize_einsums: bool
    _profiling_str: str
    _in1_dim: int
    _in2_dim: int

    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        instructions: List[tuple],
        in1_var: Optional[Union[List[float], torch.Tensor]] = None,
        in2_var: Optional[Union[List[float], torch.Tensor]] = None,
        out_var: Optional[Union[List[float], torch.Tensor]] = None,
        irrep_normalization: str = None,
        path_normalization: str = None,
        internal_weights: Optional[bool] = None,
        shared_weights: Optional[bool] = None,
        compile_left_right: bool = True,
        compile_right: bool = False,
        normalization=None,
        _specialized_code: Optional[bool] = None,
        _optimize_einsums: Optional[bool] = None
    ):
        super().__init__()

        if normalization is not None:
            warnings.warn(
                "`normalization` is deprecated. Use `irrep_normalization` instead.",
                DeprecationWarning
            )
            irrep_normalization = normalization

        if irrep_normalization is None:
            irrep_normalization = 'component'

        if path_normalization is None:
            path_normalization = 'element'

        assert irrep_normalization in ['component', 'norm', 'none']
        assert path_normalization in ['element', 'path', 'none']

        self.irreps_in1 = o3.Irreps(irreps_in1)
        self.irreps_in2 = o3.Irreps(irreps_in2)
        self.irreps_out = o3.Irreps(irreps_out)
        del irreps_in1, irreps_in2, irreps_out

        instructions = [x if len(x) == 6 else x + (1.0,) for x in instructions]
        instructions = [
            Instruction(
                i_in1=i_in1,
                i_in2=i_in2,
                i_out=i_out,
                connection_mode=connection_mode,
                has_weight=has_weight,
                path_weight=path_weight,
                path_shape={
                    'uvw': (self.irreps_in1[i_in1].mul, self.irreps_in2[i_in2].mul, self.irreps_out[i_out].mul),
                    'uvu': (self.irreps_in1[i_in1].mul, self.irreps_in2[i_in2].mul),
                    'uvv': (self.irreps_in1[i_in1].mul, self.irreps_in2[i_in2].mul),
                    'uuw': (self.irreps_in1[i_in1].mul, self.irreps_out[i_out].mul),
                    'uuu': (self.irreps_in1[i_in1].mul,),
                    'uvuv': (self.irreps_in1[i_in1].mul, self.irreps_in2[i_in2].mul),
                    'uvu<v': (self.irreps_in1[i_in1].mul * (self.irreps_in2[i_in2].mul - 1) // 2,),
                    'u<vw': (self.irreps_in1[i_in1].mul * (self.irreps_in2[i_in2].mul - 1) // 2, self.irreps_out[i_out].mul),
                }[connection_mode],
            )
            for i_in1, i_in2, i_out, connection_mode, has_weight, path_weight in instructions
        ]

        if in1_var is None:
            in1_var = [1.0 for _ in range(len(self.irreps_in1))]
        else:
            in1_var = [float(var) for var in in1_var]
            assert len(in1_var) == len(self.irreps_in1)

        if in2_var is None:
            in2_var = [1.0 for _ in range(len(self.irreps_in2))]
        else:
            in2_var = [float(var) for var in in2_var]
            assert len(in2_var) == len(self.irreps_in2)

        if out_var is None:
            out_var = [1.0 for _ in range(len(self.irreps_out))]
        else:
            out_var = [float(var) for var in out_var]
            assert len(out_var) == len(self.irreps_out)

        def num_elements(ins):
            return {
                'uvw': (self.irreps_in1[ins.i_in1].mul * self.irreps_in2[ins.i_in2].mul),
                'uvu': self.irreps_in2[ins.i_in2].mul,
                'uvv': self.irreps_in1[ins.i_in1].mul,
                'uuw': self.irreps_in1[ins.i_in1].mul,
                'uuu': 1,
                'uvuv': 1,
                'uvu<v': 1,
                'u<vw': self.irreps_in1[ins.i_in1].mul * (self.irreps_in2[ins.i_in2].mul - 1) // 2,
            }[ins.connection_mode]

        normalization_coefficients = []
        for ins in instructions:
            mul_ir_in1 = self.irreps_in1[ins.i_in1]
            mul_ir_in2 = self.irreps_in2[ins.i_in2]
            mul_ir_out = self.irreps_out[ins.i_out]
            assert mul_ir_in1.ir.p * mul_ir_in2.ir.p == mul_ir_out.ir.p
            assert abs(mul_ir_in1.ir.l - mul_ir_in2.ir.l) <= mul_ir_out.ir.l <= mul_ir_in1.ir.l + mul_ir_in2.ir.l

            if irrep_normalization == 'component':
                alpha = mul_ir_out.ir.dim
            if irrep_normalization == 'norm':
                alpha = mul_ir_in1.ir.dim * mul_ir_in2.ir.dim
            if irrep_normalization == 'none':
                alpha = 1

            if path_normalization == 'element':
                x = sum(
                    in1_var[i.i_in1] * in2_var[i.i_in2] * num_elements(i)
                    for i in instructions
                    if i.i_out == ins.i_out
                )
            if path_normalization == 'path':
                x = in1_var[ins.i_in1] * in2_var[ins.i_in2] * num_elements(ins)
                x *= len([i for i in instructions if i.i_out == ins.i_out])
            if path_normalization == 'none':
                x = 1

            if x > 0.0:
                alpha /= x

            alpha *= out_var[ins.i_out]
            alpha *= ins.path_weight

            normalization_coefficients += [sqrt(alpha)]

        self.instructions = [
            Instruction(ins.i_in1, ins.i_in2, ins.i_out, ins.connection_mode, ins.has_weight, alpha, ins.path_shape)
            for ins, alpha in zip(instructions, normalization_coefficients)
        ]

        self._in1_dim = self.irreps_in1.dim
        self._in2_dim = self.irreps_in2.dim

        if shared_weights is False and internal_weights is None:
            internal_weights = False

        if shared_weights is None:
            shared_weights = True

        if internal_weights is None:
            internal_weights = shared_weights and any(i.has_weight for i in self.instructions)

        assert shared_weights or not internal_weights
        self.internal_weights = internal_weights
        self.shared_weights = shared_weights

        opt_defaults = e3nn.get_optimization_defaults()
        self._specialized_code = _specialized_code if _specialized_code is not None else opt_defaults['specialized_code']
        self._optimize_einsums = _optimize_einsums if _optimize_einsums is not None else opt_defaults['optimize_einsums']
        del opt_defaults

        if compile_left_right:
            graphmod_left_right = codegen_tensor_product_left_right(
                self.irreps_in1,
                self.irreps_in2,
                self.irreps_out,
                self.instructions,
                self.shared_weights,
                self._specialized_code,
                self._optimize_einsums
            )
        else:
            graphmod_left_right = fx.Graph()
            graphmod_left_right.placeholder('x1', torch.Tensor)
            graphmod_left_right.placeholder('x2', torch.Tensor)
            graphmod_left_right.placeholder('w', torch.Tensor)
            graphmod_left_right.call_function(
                torch._assert,
                args=(False, "`left_right` method is not compiled")
            )
            graphmod_left_right = fx.GraphModule(torch.nn.Module(), graphmod_left_right, class_name="tp_forward")

        if compile_right:
            graphmod_right = codegen_tensor_product_right(
                self.irreps_in1,
                self.irreps_in2,
                self.irreps_out,
                self.instructions,
                self.shared_weights,
                self._specialized_code,
                self._optimize_einsums
            )
        else:
            graphmod_right = fx.Graph()
            graphmod_right.placeholder('x2', torch.Tensor)
            graphmod_right.placeholder('w', torch.Tensor)
            graphmod_right.call_function(
                torch._assert,
                args=(False, "`right` method is not compiled")
            )
            graphmod_right = fx.GraphModule(torch.nn.Module(), graphmod_right, class_name="tp_forward")

        self._codegen_register({
            "_compiled_main_left_right": graphmod_left_right,
            "_compiled_main_right": graphmod_right
        })

        self.weight_numel = sum(prod(ins.path_shape) for ins in self.instructions if ins.has_weight)

        # LoRA attributes initialization
        self.alpha = 16
        self.r = 16
        # Calculate LoRA weight numel based on path shapes
        lora_numel = 0
        for ins in self.instructions:
            if ins.has_weight:
                if len(ins.path_shape) >= 2:
                    # For 2D or 3D weight matrices
                    if len(ins.path_shape) == 3:
                        # uvw mode: flatten first two dims
                        rows = ins.path_shape[0] * ins.path_shape[1]
                        cols = ins.path_shape[2]
                    else:
                        # 2D modes
                        rows = ins.path_shape[0]
                        cols = ins.path_shape[1]
                    lora_numel += rows * self.r + self.r * cols
                else:
                    # 1D path_shape - treat as column vector
                    lora_numel += ins.path_shape[0] * self.r + self.r
        self.LoRA_weight_numel = lora_numel

        if internal_weights and self.weight_numel > 0:
            assert self.shared_weights, "Having internal weights impose shared weights"
            self.weight = torch.nn.Parameter(torch.randn(self.weight_numel))
        else:
            self.register_buffer('weight', torch.Tensor())

        if self.irreps_out.dim > 0:
            output_mask = torch.cat([
                torch.ones(mul * ir.dim)
                if any(
                    (ins.i_out == i_out) and (ins.path_weight != 0) and (0 not in ins.path_shape)
                    for ins in self.instructions
                )
                else torch.zeros(mul * ir.dim)
                for i_out, (mul, ir) in enumerate(self.irreps_out)
            ])
        else:
            output_mask = torch.ones(0)
        self.register_buffer('output_mask', output_mask)

        self._profiling_str = str(self)

    def __repr__(self):
        npath = sum(prod(i.path_shape) for i in self.instructions)
        return (
            f"{self.__class__.__name__}"
            f"({self.irreps_in1.simplify()} x {self.irreps_in2.simplify()} "
            f"-> {self.irreps_out.simplify()} | {npath} paths | {self.weight_numel} weights | {self.LoRA_weight_numel} ELoRA_weights)"
        )

    def compute_deltaW_via_svd(self):
        """Compute SVD per instruction and store top-rank factors (A, B trainable)."""
        if not hasattr(self, "weight") or self.weight.numel() == 0:
            return
        
        self.LORA_A_list = []
        self.LORA_B_list = []
        self.instruction_offsets = []
        self.S_r_list = []
        
        with torch.no_grad():
            offset = 0
            for ins_idx, ins in enumerate(self.instructions):
                if not ins.has_weight:
                    continue
                
                path_shape = ins.path_shape
                weight_size = prod(path_shape)
                W_ins = self.weight.data[offset:offset + weight_size]
                
                # Reshape based on connection mode and path_shape
                if ins.connection_mode == 'uvw' and len(path_shape) == 3:
                    # Flatten first two dimensions for matrix decomposition
                    mul_in1, mul_in2, mul_out = path_shape
                    W_matrix = W_ins.reshape(mul_in1 * mul_in2, mul_out)
                elif len(path_shape) == 2:
                    # 2D weight matrix
                    W_matrix = W_ins.reshape(path_shape)
                elif len(path_shape) == 1:
                    # 1D weight - treat as column vector
                    W_matrix = W_ins.reshape(-1, 1)
                else:
                    raise ValueError(f"Unexpected path_shape length: {len(path_shape)}")
                
                print(f"Instruction {ins_idx}: mode={ins.connection_mode}, "
                      f"path_shape={path_shape}, matrix_shape={W_matrix.shape}")
                
                # Perform SVD
                U, S, Vh = torch.linalg.svd(W_matrix, full_matrices=False)
                r = min(self.r, S.size(0))
                
                if r >= self.r:
                    print(f"rank {r}")
                    # Store components: A = U * sqrt(S), B = sqrt(S) * Vh
                    self.LORA_A_list.append(torch.nn.Parameter(U[:, :r].clone() @ torch.diag(S[:r].clone() ** 0.5)))
                    self.LORA_B_list.append(torch.nn.Parameter(torch.diag(S[:r].clone() ** 0.5) @ Vh[:r, :].clone()))
                    
                    # Store residual (not trainable)
                    self.S_r_list.append(S[r:].clone())
                    self.register_buffer(f"S_r_{ins_idx}", S[r:].clone())
                    self.register_buffer(f"W_res_{ins_idx}", U[:, r:].clone() @ torch.diag(S[r:].clone()) @ Vh[r:, :].clone())
                else:
                    print(f"rank {r} < {self.r}, using random initialization instead")
                    self.LORA_A_list.append(torch.nn.Parameter(torch.randn(U[:, :r].shape, device=U.device)))
                    self.LORA_B_list.append(torch.nn.Parameter(torch.zeros(Vh[:r, :].shape, device=Vh.device)))
                    
                    self.S_r_list.append(None)
                    self.register_buffer(f"S_r_{ins_idx}", None)
                
                self.instruction_offsets.append(offset)
                offset += weight_size
        
        self.LORA_A_list = torch.nn.ParameterList(self.LORA_A_list)
        self.LORA_B_list = torch.nn.ParameterList(self.LORA_B_list)
        print(f"[TensorProduct] SVD decomposition complete for {len(self.LORA_A_list)} instructions")

    def reconstruct_weight(self):
        """Low-rank reconstruction from A and B matrices."""
        if not hasattr(self, 'LORA_A_list'):
            return torch.zeros(self.weight_numel, device=self.weight.device, dtype=self.weight.dtype)
        
        weight_parts = []
        lora_idx = 0
        offset = 0
        
        for ins_idx, ins in enumerate(self.instructions):
            if not ins.has_weight:
                continue
            
            A_r = self.LORA_A_list[lora_idx]
            B_r = self.LORA_B_list[lora_idx]
            
            # Reconstruct weight matrix
            W_reconstructed = A_r @ B_r
            
            # Add residual if using SVD LoRA
            if hasattr(self, f"W_res_{ins_idx}"):
                res_r = getattr(self, f"W_res_{ins_idx}")
                # Get original weight for this instruction
                path_shape = ins.path_shape
                weight_size = prod(path_shape)
                W_ins = self.weight.data[offset:offset + weight_size]
                
                # Reshape to match W_reconstructed shape
                if ins.connection_mode == 'uvw' and len(path_shape) == 3:
                    mul_in1, mul_in2, mul_out = path_shape
                    W_matrix = W_ins.reshape(mul_in1 * mul_in2, mul_out)
                elif len(path_shape) == 2:
                    W_matrix = W_ins.reshape(path_shape)
                elif len(path_shape) == 1:
                    W_matrix = W_ins.reshape(-1, 1)
                else:
                    W_matrix = W_ins
                
                # Compute delta: reconstructed + residual - original
                W_reconstructed = W_reconstructed + res_r - W_matrix
            
            weight_parts.append(W_reconstructed.flatten())
            lora_idx += 1
            offset += prod(ins.path_shape)
        
        if len(weight_parts) == 0:
            return torch.zeros(self.weight_numel, device=self.weight.device, dtype=self.weight.dtype)
        
        return torch.cat(weight_parts)

    @torch.jit.unused
    def _prep_weights_python(self, weight: Optional[Union[torch.Tensor, List[torch.Tensor]]]) -> Optional[torch.Tensor]:
        if isinstance(weight, list):
            weight_shapes = [ins.path_shape for ins in self.instructions if ins.has_weight]
            if not self.shared_weights:
                weight = [w.reshape(-1, prod(shape)) for w, shape in zip(weight, weight_shapes)]
            else:
                weight = [w.reshape(prod(shape)) for w, shape in zip(weight, weight_shapes)]
            return torch.cat(weight, dim=-1)
        else:
            return weight

    def _get_weights(self, weight: Optional[torch.Tensor]) -> torch.Tensor:
        if not torch.jit.is_scripting():
            weight = self._prep_weights_python(weight)
        if weight is None:
            if self.weight_numel > 0 and not self.internal_weights:
                raise RuntimeError("Weights must be provided when internal_weights = False")
            weight = self.weight
            # Apply low-rank adaptation if SVD decomposition exists
            if hasattr(self, "LORA_A_list"):
                delta_weight = self.reconstruct_weight()
                weight = weight + delta_weight
        if weight is not None:
            if self.shared_weights:
                assert weight.shape == (self.weight_numel,), "Invalid weight shape"
            else:
                assert weight.shape[-1] == self.weight_numel, "Invalid weight shape"
                assert weight.ndim > 1, "When shared weights is false, weights must have batch dimension"
        return weight

    @torch.jit.export
    def right(self, y, weight: Optional[torch.Tensor] = None):
        r"""Partially evaluate w x ⊗ y."""
        assert y.shape[-1] == self._in2_dim, "Incorrect last dimension for y"
        real_weight = self._get_weights(weight)
        return self._compiled_main_right(y, real_weight)

    def forward(self, x, y, weight: Optional[torch.Tensor] = None):
        r"""Evaluate w x ⊗ y."""
        assert x.shape[-1] == self._in1_dim, "Incorrect last dimension for x"
        assert y.shape[-1] == self._in2_dim, "Incorrect last dimension for y"
        real_weight = self._get_weights(weight)
        return self._compiled_main_left_right(x, y, real_weight)

    def weight_view_for_instruction(
        self,
        instruction: int,
        weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        r"""View of weights corresponding to instruction."""
        if not self.instructions[instruction].has_weight:
            raise ValueError(f"Instruction {instruction} has no weights.")
        offset = sum(prod(ins.path_shape) for ins in self.instructions[:instruction] if ins.has_weight)
        ins = self.instructions[instruction]
        weight = self._get_weights(weight)
        batchshape = weight.shape[:-1]
        return weight.narrow(-1, offset, prod(ins.path_shape)).view(batchshape + ins.path_shape)

    def weight_views(
        self,
        weight: Optional[torch.Tensor] = None,
        yield_instruction: bool = False
    ):
        r"""Iterator over weight views for each weighted instruction."""
        weight = self._get_weights(weight)
        batchshape = weight.shape[:-1]
        offset = 0
        for ins_i, ins in enumerate(self.instructions):
            if ins.has_weight:
                flatsize = prod(ins.path_shape)
                this_weight = weight.narrow(-1, offset, flatsize).view(batchshape + ins.path_shape)
                offset += flatsize
                if yield_instruction:
                    yield ins_i, ins, this_weight
                else:
                    yield this_weight

    def merge_LoRA(self):
        """Merge the low-rank SVD updates back into the main weight."""
        if not hasattr(self, 'LORA_A_list'):
            return
        
        # Add reconstructed delta to original weight
        self.weight.data = self.weight.data + self.reconstruct_weight()
        
        # Clean up all SVD components
        for ins_idx, ins in enumerate(self.instructions):
            if ins.has_weight:
                # Delete buffers
                if hasattr(self, f"S_r_{ins_idx}"):
                    delattr(self, f"S_r_{ins_idx}")
                if hasattr(self, f"W_res_{ins_idx}"):
                    delattr(self, f"W_res_{ins_idx}")
        
        # Clean up lists and attributes
        if hasattr(self, 'LORA_A_list'):
            del self.LORA_A_list
        if hasattr(self, 'LORA_B_list'):
            del self.LORA_B_list
        if hasattr(self, 'S_r_list'):
            del self.S_r_list
        if hasattr(self, 'instruction_offsets'):
            del self.instruction_offsets
        if hasattr(self, 'alpha'):
            del self.alpha
        if hasattr(self, 'r'):
            del self.r
        if hasattr(self, 'LoRA_weight_numel'):
            del self.LoRA_weight_numel