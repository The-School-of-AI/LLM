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
    parser.add_argument(
        "--reduce-bucket-size",
        type=int,
        default=None,
        help="Override DeepSpeed ZeRO reduce_bucket_size at runtime.",
    )
    parser.add_argument(
        "--max-steps-per-epoch",
        type=int,
        default=None,
        help="Override training.max_steps_per_epoch for short tuning runs.",
    )
    parser.add_argument(
        "--max-val-steps",
        type=int,
        default=None,
        help="Override training.max_val_steps for short tuning runs.",
    )
    parser.add_argument(
        "--profile-steps",
        nargs="*",
        type=int,
        default=None,
        help="Override training.profile_steps with an explicit list of steps.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override config.output_dir for this run.",
    )
    parser.add_argument(
        "--run-id-suffix",
        type=str,
        default=None,
        help="Append a suffix to the resolved run_id for this run.",
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

    if args.reduce_bucket_size is not None:
        cfg.training.reduce_bucket_size = args.reduce_bucket_size
    if args.max_steps_per_epoch is not None:
        cfg.training.max_steps_per_epoch = args.max_steps_per_epoch
    if args.max_val_steps is not None:
        cfg.training.max_val_steps = args.max_val_steps
    if args.profile_steps is not None:
        cfg.training.profile_steps = args.profile_steps
    if args.output_dir is not None:
        cfg.output_dir = os.path.abspath(args.output_dir)
    if args.run_id_suffix:
        cfg.run_id = f"{cfg.run_id}_{args.run_id_suffix}"

    try:
        main(args.local_rank, cfg)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
