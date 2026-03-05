import os

import torch.distributed as dist
from jsonargparse import ArgumentParser

from llm.config import Config
from llm.pretrainer import PreTrainer


def main(local_rank: int, c: Config):
    pretrainer = PreTrainer(local_rank, c)
    pretrainer.run()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_class_arguments(Config, "config", required=True)

    # DeepSpeed / torchrun injects --local_rank into argv. Accept it here so
    # jsonargparse doesn't error on an unknown argument. The actual value is
    # always read from os.environ inside the training code.
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank injected by the DeepSpeed / torchrun launcher. "
        "Do not set manually.",
    )

    args = parser.parse_args()
    if not hasattr(args.config, "__path__") or args.config.__path__ is None:
        parser.error("--config: a path to a YAML config file is required")

    cfg: Config = parser.instantiate_classes(args).config

    # Resolve all relative paths against the config file's directory so that
    # paths in the YAML work regardless of where the script is invoked from.
    config_path = args.config.__path__.absolute
    config_dir = os.path.dirname(os.path.abspath(config_path)) if config_path else None
    cfg.resolve_paths(base_dir=config_dir)

    try:
        main(args.local_rank, cfg)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
