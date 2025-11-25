import torch
from transformers.modeling_outputs import CausalLMOutputWithPast


def unwrap_backbone(model):
    """
    递归解包PEFT/Distributed包装，返回真正的backbone（如Qwen3VLModel）。
    """
    current = model
    visited = set()

    while True:
        next_candidate = None

        # 优先处理PEFT的base_model
        if hasattr(current, "base_model") and current.base_model is not None:
            next_candidate = current.base_model

        # transformers通常将backbone挂在.model上
        elif hasattr(current, "model") and current.model is not None and current.model is not current:
            next_candidate = current.model

        # DDP / Accelerate 可能将真实模型包在.module
        elif hasattr(current, "module") and current.module is not None:
            next_candidate = current.module

        if next_candidate is None or id(next_candidate) in visited:
            break

        visited.add(id(next_candidate))
        current = next_candidate

    return current


def forward_backbone(model, *, return_dict=True, use_cache=False, output_hidden_states=False, **kwargs):
    """
    对backbone执行forward，确保返回BaseModelOutput系列并包含last_hidden_state。
    """
    backbone = unwrap_backbone(model)
    return backbone(
        return_dict=return_dict,
        use_cache=use_cache,
        output_hidden_states=output_hidden_states,
        **kwargs,
    )


def ensure_last_hidden_state(outputs):
    last_hidden_state = getattr(outputs, "last_hidden_state", None)
    if last_hidden_state is None:
        raise RuntimeError("backbone outputs missing last_hidden_state")
    return last_hidden_state


def build_causal_lm_output(model, backbone_outputs, *, hidden_states=None):
    """
    根据backbone输出构造CausalLMOutputWithPast，并附带最后一层hidden state。
    """
    if hidden_states is None:
        hidden_states = ensure_last_hidden_state(backbone_outputs)

    lm_input = hidden_states.to(model.lm_head.weight.dtype)
    logits = model.lm_head(lm_input)

    outputs = CausalLMOutputWithPast(
        logits=logits,
        past_key_values=backbone_outputs.past_key_values,
        hidden_states=None,
        attentions=None,
    )
    # 额外挂载last_hidden_state，方便后续使用
    outputs.last_hidden_state = hidden_states
    return outputs


