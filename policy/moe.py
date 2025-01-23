from typing import Dict, List, Optional, Union

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig

from .base import Policy


class MoE(Policy):
    def __init__(
        self,
        base_params,
        decomposed_params,
        base_policy_cfg: Optional[Union[DictConfig, int]],
        params_paths: List[str],
        gpu,
        norm_coeffs,
        per_layer,
        init_values: Optional[List[float]] = None,
        **kwargs,
    ):
        # Create learnable parameters.
        nn.Module.__init__(self=self)
        weights_dict_list: List[Dict[str, torch.Tensor]] = []
        if base_policy_cfg is None:
            base_policy = Policy(base_params=base_params, gpu=gpu, init_val=0)
        elif isinstance(base_policy_cfg, DictConfig):
            base_policy: Policy = hydra.utils.instantiate(
                base_policy_cfg,
                base_params=base_params,
                decomposed_params=decomposed_params,
                gpu=gpu,
            )
        else:
            raise NotImplementedError
        with torch.no_grad():
            for i, load_ckpt in enumerate(params_paths):
                print(f"Loading checkpoint {i} at {load_ckpt}...")
                if "learnable_params" in load_ckpt:
                    learnable_params = torch.load(load_ckpt)
                else:
                    state_dict = torch.load(load_ckpt, weights_only=True)
                    base_policy.load_state_dict(state_dict=state_dict)
                    learnable_params = base_policy.get_learnable_params()
                weights_dict_list.append(
                    {k: torch.detach_copy(p) for k, p in learnable_params.items()}
                )

        self.num_weights_dict = len(weights_dict_list)

        self.num_params_per_weight_dict = 0
        for _ in weights_dict_list[0]:
            self.num_params_per_weight_dict += 1

        self.num_params = self.num_weights_dict * self.num_params_per_weight_dict
        if init_values is None:
            init_values = torch.Tensor(
                [1 / self.num_weights_dict for _ in range(self.num_weights_dict)]
            ) # shape (3,)
        else:
            assert len(init_values) == self.num_weights_dict
            init_values = torch.Tensor(init_values)
        self.learned_params_per_weight_dict = 1
        if per_layer:
            self.learned_params_per_weight_dict = self.num_params_per_weight_dict
        init_values = torch.stack(
            [init_values for _ in range(self.learned_params_per_weight_dict)], dim=1
        ) # shape (3, 1/num_layers)
        if norm_coeffs:
            # Normalize across different weight idxs (for all layers)
            init_values = init_values / torch.sum(init_values, axis=0)

        # Num weight idxs x learned params_per_weight_idx
        self.adaptive_weights = torch.nn.Parameter(
            data=init_values,
            requires_grad=True,
        ) # shape (3, 1/num_layers)

        self.parameter_keys = []
        self.original_params = {}
        for k, v in weights_dict_list[0].items():
            self.parameter_keys.append(k)
            self.original_params[k] = []

            for i, weight_dict in enumerate(weights_dict_list):
                weight_tensor = self.get_mask(p=weight_dict[k])
                new_key = k.replace(".", "_")
                self.register_buffer(
                    f"weights_{i}_k_{new_key}",
                    tensor=weight_tensor,
                )
                if isinstance(gpu, DictConfig):
                    self.original_params[k].append(weight_tensor.to(device=gpu[k]))
                else:
                    self.original_params[k].append(weight_tensor.to(device=gpu))

        self.norm = norm_coeffs
        self.per_layer = per_layer
        self.trainable_params = [self.adaptive_weights]

    def get_weight_to_combine(self, k, weights_dict_idx):
        new_key = k.replace(".", "_")
        return getattr(self, f"weights_{weights_dict_idx}_k_{new_key}")

    def get_coeff_per_layer(self):
        if self.norm:
            adaptive_weights = self.adaptive_weights / self.adaptive_weights.sum(0)
        else:
            adaptive_weights = self.adaptive_weights
        weights_per_layer = adaptive_weights.expand(
            [
                self.num_weights_dict,
                self.num_params_per_weight_dict,
            ]
        )
        return weights_per_layer

    def get_learnable_params(self):
        adaptive_coeff_per_layer = self.get_coeff_per_layer()
        output_params = {}
        for i, (k, vs) in enumerate(self.original_params.items()):
            cs_coeff = adaptive_coeff_per_layer[:, i]
            out = vs[0] * cs_coeff[0]
            for j, other_v in enumerate(vs[1:]):
                v_idx = j + 1
                out = out + other_v * cs_coeff[v_idx]
            output_params[k] = out
        return output_params

    def get_mask(self, p):
        return p

    def record_state(self, metrics_to_log):
        avg_weights = self.adaptive_weights.mean(-1).detach().cpu().numpy()
        dict_to_log = {
            f"adaptive_weight/mean_across_params_w{i}": w
            for i, w in enumerate(avg_weights.tolist())
        }
        metrics_to_log.update(**dict_to_log)
