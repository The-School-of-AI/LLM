from pathlib import Path

import torch.distributed as dist
from exp.train import Config, Trainer
from jsonargparse import ArgumentParser


def resolve_paths(c: Config, config_file_path: Path | str) -> Config:
    if isinstance(config_file_path, str):
        config_file_path = Path(config_file_path)

    def _resolve_path(path: str) -> str:
        return str((config_file_path.parent / path).resolve())

    c.deepspeed_config = _resolve_path(c.deepspeed_config)
    c.tokenizer_dir = _resolve_path(c.tokenizer_dir)
    c.profiler_output_dir = _resolve_path(c.profiler_output_dir)

    return c


def main(local_rank: int, config: Config):
    trainer = Trainer(local_rank, config)
    trainer.train()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)  # set by deepspeed
    parser.add_class_arguments(Config, "config")

    args = parser.parse_args()
    clss = parser.instantiate_classes(args)
    config = resolve_paths(clss.config, args.config.__path__.absolute)

    try:
        main(args.local_rank, config)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
