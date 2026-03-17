#!/usr/bin/env python3
"""
SFT Preprocessing Pipeline — CLI
Team 18 / sft_training_new

Usage:
    python cli.py --config config/pipeline.yaml run-all
    python cli.py --config config/pipeline.yaml mix
    python cli.py --config config/pipeline.yaml validate
    python cli.py --config config/pipeline.yaml clean
    python cli.py --config config/pipeline.yaml template
    python cli.py --config config/pipeline.yaml tokenize
    python cli.py --config config/pipeline.yaml mask
    python cli.py --config config/pipeline.yaml quality
    python cli.py --config config/pipeline.yaml run-all --resume-from-stage 3
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from pipeline.config import PipelineConfig
from pipeline.funnel_tracker import FunnelTracker


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
        stream=sys.stdout,
    )


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.option(
    "--config", "-c",
    default="config/pipeline.yaml",
    show_default=True,
    help="Path to pipeline YAML config.",
)
@click.option(
    "--work-dir",
    default=None,
    help="Override global.work_dir from config.",
)
@click.option(
    "--output-dir",
    default=None,
    help="Override global.output_dir from config.",
)
@click.option(
    "--log-level",
    default=None,
    help="Override global.log_level (DEBUG|INFO|WARNING|ERROR).",
)
@click.pass_context
def cli(ctx: click.Context, config: str, work_dir: str | None,
        output_dir: str | None, log_level: str | None) -> None:
    """SFT Preprocessing Pipeline — Team 18."""
    cfg = PipelineConfig.from_yaml(config)
    if work_dir:
        cfg.globals.work_dir = work_dir
    if output_dir:
        cfg.globals.output_dir = output_dir
    if log_level:
        cfg.globals.log_level = log_level

    _setup_logging(cfg.globals.log_level)

    # Create work/output dirs upfront
    Path(cfg.globals.work_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.globals.output_dir).mkdir(parents=True, exist_ok=True)

    ctx.ensure_object(dict)
    ctx.obj["cfg"] = cfg


# ---------------------------------------------------------------------------
# Helper: count stage0 output for FunnelTracker init
# ---------------------------------------------------------------------------

def _build_tracker(cfg: PipelineConfig) -> FunnelTracker:
    from pipeline.io import readers
    funnel_path = Path(cfg.globals.output_dir) / "funnel_report.json"
    # Use 0 as placeholder; actual count set by stage0 run
    return FunnelTracker(total_input=0, output_path=funnel_path)


# ---------------------------------------------------------------------------
# Stage subcommands
# ---------------------------------------------------------------------------

@cli.command("mix")
@click.pass_context
def cmd_mix(ctx: click.Context) -> None:
    """Stage 0: data mixing and weighted sampling."""
    from pipeline import stage0_mix
    cfg = ctx.obj["cfg"]
    tracker = _build_tracker(cfg)
    stage0_mix.run(cfg, tracker)
    tracker.save()


@cli.command("validate")
@click.pass_context
def cmd_validate(ctx: click.Context) -> None:
    """Stage 1: schema validation."""
    from pipeline import stage1_validate
    cfg = ctx.obj["cfg"]
    tracker = _build_tracker(cfg)
    stage1_validate.run(cfg, tracker)
    tracker.save()


@cli.command("clean")
@click.pass_context
def cmd_clean(ctx: click.Context) -> None:
    """Stage 2: cleaning and filtering."""
    from pipeline import stage2_clean
    cfg = ctx.obj["cfg"]
    tracker = _build_tracker(cfg)
    stage2_clean.run(cfg, tracker)
    tracker.save()


@cli.command("template")
@click.pass_context
def cmd_template(ctx: click.Context) -> None:
    """Stage 3: chat template application."""
    from pipeline import stage3_template
    cfg = ctx.obj["cfg"]
    tracker = _build_tracker(cfg)
    stage3_template.run(cfg, tracker)
    tracker.save()


@cli.command("tokenize")
@click.pass_context
def cmd_tokenize(ctx: click.Context) -> None:
    """Stage 4: tokenization."""
    from pipeline import stage4_tokenize
    cfg = ctx.obj["cfg"]
    tracker = _build_tracker(cfg)
    stage4_tokenize.run(cfg, tracker)
    tracker.save()


@cli.command("mask")
@click.pass_context
def cmd_mask(ctx: click.Context) -> None:
    """Stage 5: loss masking."""
    from pipeline import stage5_mask
    cfg = ctx.obj["cfg"]
    tracker = _build_tracker(cfg)
    stage5_mask.run(cfg, tracker)
    tracker.save()


@cli.command("quality")
@click.pass_context
def cmd_quality(ctx: click.Context) -> None:
    """Stage 6: quality validation, sharding, and (optional) S3 upload."""
    from pipeline import stage6_quality
    cfg = ctx.obj["cfg"]
    tracker = _build_tracker(cfg)
    stage6_quality.run(cfg, tracker)
    tracker.save()


# ---------------------------------------------------------------------------
# run-all
# ---------------------------------------------------------------------------

@cli.command("run-all")
@click.option(
    "--resume-from-stage", "resume_from",
    type=int,
    default=None,
    help="Skip stages before this number (0=start over). Overrides global.resume_from_stage.",
)
@click.pass_context
def cmd_run_all(ctx: click.Context, resume_from: int | None) -> None:
    """Run the full pipeline (stages 0–6) in order."""
    from pipeline import (
        stage0_mix,
        stage1_validate,
        stage2_clean,
        stage3_template,
        stage4_tokenize,
        stage5_mask,
        stage6_quality,
    )

    cfg = ctx.obj["cfg"]
    if resume_from is not None:
        cfg.globals.resume_from_stage = resume_from

    start = cfg.globals.resume_from_stage
    if start > 0:
        logging.getLogger(__name__).info(
            "Resuming from stage %d (stages 0–%d will be skipped if output exists)",
            start, start - 1,
        )

    funnel_path = Path(cfg.globals.output_dir) / "funnel_report.json"
    tracker = FunnelTracker(total_input=0, output_path=funnel_path)

    stages = [
        (0, "mix",      stage0_mix),
        (1, "validate", stage1_validate),
        (2, "clean",    stage2_clean),
        (3, "template", stage3_template),
        (4, "tokenize", stage4_tokenize),
        (5, "mask",     stage5_mask),
        (6, "quality",  stage6_quality),
    ]

    log = logging.getLogger(__name__)
    for stage_num, stage_name, module in stages:
        if stage_num < start:
            log.info("Skipping stage %d (%s) — resume_from_stage=%d", stage_num, stage_name, start)
            # Verify intermediate file exists when skipping
            if stage_num < 6:
                expected = cfg.work_path(
                    _stage_output_file(cfg, stage_num)
                )
                if not expected.exists():
                    log.error(
                        "Cannot resume from stage %d: expected intermediate file %s does not exist",
                        start, expected,
                    )
                    sys.exit(1)
            continue

        log.info("=" * 60)
        log.info("Running stage %d: %s", stage_num, stage_name)
        log.info("=" * 60)
        module.run(cfg, tracker)

    tracker.save()
    log.info("Pipeline complete.")


def _stage_output_file(cfg: PipelineConfig, stage_num: int) -> str:
    mapping = {
        0: cfg.stage0.output_file,
        1: cfg.stage1.output_file,
        2: cfg.stage2.output_file,
        3: cfg.stage3.output_file,
        4: cfg.stage4.output_file,
        5: cfg.stage5.output_file,
    }
    return mapping.get(stage_num, "")


if __name__ == "__main__":
    cli()
