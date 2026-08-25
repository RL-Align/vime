from types import SimpleNamespace

import torch

from vime.utils.train_dump_utils import save_debug_train_data


def test_debug_dump_adds_rank_and_replaces_atomically(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 2)
    args = SimpleNamespace(save_debug_train_data=str(tmp_path / "{rollout_id}.pt"))

    save_debug_train_data(
        args,
        rollout_id=5,
        rollout_data={"log_probs": [torch.tensor([1.0])]},
    )

    output = tmp_path / "5.rank2.pt"
    payload = torch.load(output, weights_only=True)
    assert payload["rollout_id"] == 5
    assert payload["rank"] == 2
    torch.testing.assert_close(payload["rollout_data"]["log_probs"][0], torch.tensor([1.0]))
    assert list(tmp_path.glob(".*.tmp")) == []
