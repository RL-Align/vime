from argparse import Namespace

import _cp_dist_helpers  # noqa: F401
import pytest
import torch
from megatron.core import mpu

from vime.backends.megatron_utils import loss as loss_module
from vime.backends.megatron_utils.loss import (
    _build_topp_keep_mask,
    get_log_probs_and_entropy,
    get_rollout_top_p_logprob_kwargs,
)

NUM_GPUS = 0


@pytest.mark.unit
def test_missing_top_p_replay_data_raises():
    with pytest.raises(ValueError, match="requires rollout_top_p_token_ids"):
        get_rollout_top_p_logprob_kwargs(Namespace(rollout_top_p=0.95), {})


def _set_cp(monkeypatch, *, size: int, rank: int) -> None:
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: size)
    monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda: rank)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_rank", lambda: 0, raising=False)


def _kept_ids(row: torch.Tensor) -> list[int]:
    return row.nonzero(as_tuple=False).squeeze(-1).tolist()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("rank", "expected"),
    [
        (0, {2: [107]}),
        (1, {1: [104], 2: [105], 3: [106]}),
    ],
)
def test_top_p_mask_aligns_with_zigzag_cp_response_rows(monkeypatch, rank, expected):
    _set_cp(monkeypatch, size=2, rank=rank)
    keep = _build_topp_keep_mask(
        4,
        200,
        torch.device("cpu"),
        top_p_token_ids=[[104, 105, 106, 107]],
        top_p_token_offsets=[[0, 1, 2, 3, 4]],
        total_lengths=[8],
        response_lengths=[4],
        allgather_cp=False,
    )

    masked_rows = {row: _kept_ids(keep[row]) for row in range(keep.size(0)) if not keep[row].all()}
    assert masked_rows == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    ("rank", "expected"),
    [
        (0, {1: [102], 2: [103]}),
        (1, {0: [104], 1: [105]}),
    ],
)
def test_top_p_mask_aligns_with_allgather_cp_response_rows(monkeypatch, rank, expected):
    _set_cp(monkeypatch, size=2, rank=rank)
    keep = _build_topp_keep_mask(
        3,
        200,
        torch.device("cpu"),
        top_p_token_ids=[[102, 103, 104, 105]],
        top_p_token_offsets=[[0, 1, 2, 3, 4]],
        total_lengths=[6],
        response_lengths=[4],
        allgather_cp=True,
    )

    masked_rows = {row: _kept_ids(keep[row]) for row in range(keep.size(0)) if not keep[row].all()}
    assert masked_rows == expected


@pytest.mark.unit
def test_top_p_mask_aligns_with_cp1_response_rows(monkeypatch):
    _set_cp(monkeypatch, size=1, rank=0)
    keep = _build_topp_keep_mask(
        9,
        30,
        torch.device("cpu"),
        top_p_token_ids=[[13, 99, 14], [21, 22, 99, 23]],
        top_p_token_offsets=[[0, 2, 3], [0, 1, 3, 4]],
        total_lengths=[5, 4],
        response_lengths=[2, 3],
        allgather_cp=False,
    )

    masked_rows = {row: _kept_ids(keep[row]) for row in range(keep.size(0)) if not keep[row].all()}
    assert masked_rows == {2: [13], 3: [14], 5: [21], 6: [22], 7: [23]}


@pytest.mark.unit
def test_provider_receives_unscaled_logits_and_native_fallback_scales_once(monkeypatch):
    _set_cp(monkeypatch, size=1, rank=0)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_group", lambda: None, raising=False)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_world_size", lambda: 1, raising=False)
    observed = {}

    def calculate(logits, *_args, with_entropy, **_kwargs):
        observed["native_logits"] = logits.clone()
        return logits[:, :1], logits.sum(dim=-1) if with_entropy else None

    def dispatch(*, request, native, **_kwargs):
        observed["request_logits"] = request.logits.clone()
        observed["temperature"] = request.temperature
        return native(
            request.logits,
            request.target_ids,
            request.tensor_parallel_group,
            with_entropy=request.with_entropy,
            with_entropy_grad=request.with_entropy_grad,
            chunk_size=request.chunk_size,
            log_prob_keep_mask=request.log_prob_keep_mask,
        )

    monkeypatch.setattr(loss_module, "calculate_log_probs_and_entropy", calculate)
    monkeypatch.setattr(loss_module, "compute_linear_logp", dispatch)
    logits = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    args = Namespace(
        allgather_cp=False,
        entropy_coef=0.0,
        log_probs_chunk_size=-1,
        rollout_temperature=0.5,
    )

    get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=[torch.tensor([0, 1, 2])],
        total_lengths=[3],
        response_lengths=[2],
    )

    torch.testing.assert_close(observed["request_logits"], logits.squeeze(0))
    torch.testing.assert_close(observed["native_logits"], logits.squeeze(0) / 0.5)
    assert observed["temperature"] == 0.5


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
