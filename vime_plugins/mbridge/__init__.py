from .deepseek_v32 import DeepseekV32Bridge
from .glm4 import GLM4Bridge
from .glm4moe import GLM4MoEBridge
from .glm4moe_lite import GLM4MoELiteBridge
from .gpt_oss import GptOssBridge
from .mimo import MimoBridge
from .minimax_m2 import MiniMaxM2Bridge
from .qwen3_5 import Qwen3_5Bridge
from .qwen3_next import Qwen3NextBridge
from mbridge.models.qwen2moe import Qwen2MoEBridge
from mbridge.models.qwen3moe import Qwen3MoEBridge
import torch


_QWEN_MOE_LOCAL_NORM_MAPPING = {
    "input_layernorm.weight": ["model.layers.{layer_number}.input_layernorm.weight"],
}

for _bridge_cls in (Qwen2MoEBridge, Qwen3MoEBridge):
    _other_mapping = dict(getattr(_bridge_cls, "_OTHER_MAPPING", {}))
    _other_mapping.update(_QWEN_MOE_LOCAL_NORM_MAPPING)
    _bridge_cls._OTHER_MAPPING = _other_mapping

_ORIGINAL_QWEN_MOE_WEIGHT_NAME_MAPPING_MLP = Qwen2MoEBridge._weight_name_mapping_mlp
_ORIGINAL_QWEN_MOE_WEIGHT_TO_MCORE_FORMAT = Qwen2MoEBridge._weight_to_mcore_format


def _qwen_moe_weight_name_mapping_mlp(self, name: str) -> list[str]:
    layer_number = name.split(".")[2]
    num_experts = self.hf_config.num_experts

    if "mlp.experts.weight1" in name:
        hf_names = []
        for expert_id in range(num_experts):
            hf_names.extend(
                [
                    f"model.layers.{layer_number}.mlp.experts.{expert_id}.gate_proj.weight",
                    f"model.layers.{layer_number}.mlp.experts.{expert_id}.up_proj.weight",
                ]
            )
        return hf_names

    if "mlp.experts.weight2" in name:
        return [
            f"model.layers.{layer_number}.mlp.experts.{expert_id}.down_proj.weight"
            for expert_id in range(num_experts)
        ]

    return _ORIGINAL_QWEN_MOE_WEIGHT_NAME_MAPPING_MLP(self, name)


def _qwen_moe_weight_to_mcore_format(self, mcore_weights_name: str, hf_weights: list):
    if "mlp.experts.weight1" in mcore_weights_name:
        assert len(hf_weights) == self.hf_config.num_experts * 2
        gates = hf_weights[0::2]
        ups = hf_weights[1::2]
        return (
            torch.cat((torch.stack(gates), torch.stack(ups)), dim=-2)
            .transpose(-1, -2)
            .reshape(self.hf_config.hidden_size, -1)
            .contiguous()
        )

    if "mlp.experts.weight2" in mcore_weights_name:
        assert len(hf_weights) == self.hf_config.num_experts
        return (
            torch.stack(hf_weights)
            .transpose(-1, -2)
            .reshape(-1, self.hf_config.hidden_size)
            .contiguous()
        )

    return _ORIGINAL_QWEN_MOE_WEIGHT_TO_MCORE_FORMAT(self, mcore_weights_name, hf_weights)


for _bridge_cls in (Qwen2MoEBridge, Qwen3MoEBridge):
    _bridge_cls._weight_name_mapping_mlp = _qwen_moe_weight_name_mapping_mlp
    _bridge_cls._weight_to_mcore_format = _qwen_moe_weight_to_mcore_format

__all__ = [
    "GLM4Bridge",
    "GLM4MoEBridge",
    "GLM4MoELiteBridge",
    "GptOssBridge",
    "MiniMaxM2Bridge",
    "Qwen3NextBridge",
    "Qwen3_5Bridge",
    "MimoBridge",
    "DeepseekV32Bridge",
    "Qwen2MoEBridge",
    "Qwen3MoEBridge",
]
