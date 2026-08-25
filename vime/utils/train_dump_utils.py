import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def save_debug_train_data(args, *, rollout_id, rollout_data):
    if (path_template := args.save_debug_train_data) is not None:
        rank = torch.distributed.get_rank()
        rendered = path_template.format(rollout_id=rollout_id, rank=rank)
        path = Path(rendered)
        if "{rank}" not in path_template:
            path = path.with_name(f"{path.stem}.rank{rank}{path.suffix}")
        logger.info(f"Save debug train data to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        torch.save(
            dict(
                rollout_id=rollout_id,
                rank=rank,
                rollout_data=rollout_data,
            ),
            temporary,
        )
        os.replace(temporary, path)
