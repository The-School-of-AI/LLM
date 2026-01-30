#!/usr/bin/env python3
"""
MoE Tools CLI
=============
Command-line interface for MoE architecture tooling.

Commands:
    estimate    - Run FLOPs/memory/parameter estimation
    profile     - Run training profiler
    diagnose    - Run routing diagnostics
    dashboard   - Generate dashboard configuration

Usage:
    python -m moe_tools estimate 70b_moe
    python -m moe_tools profile --config config.json
    python -m moe_tools diagnose --export metrics.json
    python -m moe_tools dashboard --export dashboard.json
    python -m moe_tools all 70b_moe --output report.json
"""

import argparse
import json
import sys
from pathlib import Path


def run_estimate(args):
    """Run estimation tools."""
    from estimators.flops_estimator import FLOPEstimator, CONFIGS as FLOP_CONFIGS
    from estimators.memory_estimator import MemoryEstimator, get_config as get_mem_config
    from estimators.param_counter import ParamCounter, CONFIGS as PARAM_CONFIGS
    
    model = args.model
    
    print(f"\n{'='*60}")
    print(f"MoE ESTIMATION REPORT: {model.upper()}")
    print(f"{'='*60}")
    
    # Parameter Count
    if model in PARAM_CONFIGS:
        print("\n📦 PARAMETER COUNT")
        print("-" * 40)
        counter = ParamCounter(PARAM_CONFIGS[model])
        report = counter.full_report()
        print(f"  Total Parameters: {report['summary']['total_params']}")
        print(f"  Active Parameters: {report['summary']['active_params']}")
        print(f"  Sparsity: {report['summary']['sparsity']}")
    
    # FLOPs Estimation
    if model in FLOP_CONFIGS:
        print("\n⚡ FLOPS ESTIMATION")
        print("-" * 40)
        flop_est = FLOPEstimator(FLOP_CONFIGS[model])
        flop_report = flop_est.full_report()
        print(f"  Forward FLOPs/token: {flop_report['totals']['forward_per_token']}")
        print(f"  Total FLOPs/token: {flop_report['totals']['total_per_token']}")
        print(f"  Tokens/day estimate: {flop_report['throughput']['tokens_per_day']}")
    
    # Memory Estimation
    model_cfg, dist_cfg = get_mem_config(model)
    print("\n💾 MEMORY ESTIMATION")
    print("-" * 40)
    mem_est = MemoryEstimator(model_cfg, dist_cfg)
    mem_report = mem_est.full_report()
    print(f"  Total Weights: {mem_report['weights_breakdown']['total']}")
    print(f"  Per-GPU Memory: {mem_report['distributed']['total_per_gpu']}")
    print(f"  Cluster Memory: {mem_report['distributed']['cluster_total']}")
    
    # Export if requested
    if args.output:
        combined_report = {
            'model': model,
            'parameters': report if 'report' in dir() else {},
            'flops': flop_report if 'flop_report' in dir() else {},
            'memory': mem_report,
        }
        with open(args.output, 'w') as f:
            json.dump(combined_report, f, indent=2)
        print(f"\n📄 Report exported to: {args.output}")


def run_profile(args):
    """Run training profiler."""
    from profilers.training_profiler import TrainingProfiler, ProfilerConfig
    
    config = ProfilerConfig(
        profile_every_n_steps=args.interval,
        warmup_steps=args.warmup,
        use_tensorboard=args.tensorboard,
    )
    
    profiler = TrainingProfiler(config)
    
    print("\n📊 Training Profiler Configuration")
    print("-" * 40)
    print(f"  Profile interval: every {config.profile_every_n_steps} steps")
    print(f"  Warmup: {config.warmup_steps} steps")
    print(f"  TensorBoard: {'Enabled' if config.use_tensorboard else 'Disabled'}")
    
    print("\n💡 Integration Example:")
    print("""
    from moe_tools.profilers import TrainingProfiler, ProfilerConfig
    
    profiler = TrainingProfiler(ProfilerConfig())
    
    for step in range(num_steps):
        with profiler.profile_step(batch_size=8, seq_length=2048):
            with profiler.time_region('forward'):
                output = model(input)
            with profiler.time_region('backward'):
                loss.backward()
            with profiler.time_region('optimizer'):
                optimizer.step()
        
        profiler.log_metrics()
    
    profiler.print_summary()
    """)


def run_diagnose(args):
    """Run routing diagnostics."""
    from diagnostics.routing_diagnostics import create_diagnostics
    
    model = args.model or '70b_moe'
    diagnostics = create_diagnostics(model)
    
    print("\n🔍 Routing Diagnostics Configuration")
    print("-" * 40)
    print(f"  Model: {model}")
    print(f"  Routed Experts: {diagnostics.config.num_routed_experts}")
    print(f"  Null Experts: {diagnostics.config.num_null_experts}")
    
    print("\n📊 Health Gate Thresholds:")
    print(f"  Junk→Null Rate: {diagnostics.config.min_null_junk_rate:.0%} - {diagnostics.config.max_null_junk_rate:.0%}")
    print(f"  Signal→Null Max: {diagnostics.config.max_null_signal_rate:.0%}")
    print(f"  Min Entropy: {diagnostics.config.min_routing_entropy}")
    print(f"  Max Gini: {diagnostics.config.max_gini_coefficient}")
    
    print("\n💡 Integration Example:")
    print("""
    from moe_tools.diagnostics import create_diagnostics
    
    diagnostics = create_diagnostics('70b_moe')
    
    # In training loop, for each layer:
    diagnostics.log_batch(
        layer_idx=layer_idx,
        expert_indices=expert_indices,
        expert_weights=expert_weights,
        token_ids=token_ids,
    )
    
    # At end of step:
    snapshot = diagnostics.step()
    metrics = diagnostics.get_dashboard_metrics()
    """)
    
    if args.export:
        diagnostics.export_telemetry(args.export)
        print(f"\n📄 Telemetry config exported to: {args.export}")


def run_dashboard(args):
    """Generate dashboard configuration."""
    from dashboards.team7_dashboard import Team7Dashboard
    
    dashboard = Team7Dashboard()
    
    print("\n📊 Team 7 Dashboard Configuration")
    print("-" * 40)
    
    dashboard.print_summary()
    
    if args.export:
        dashboard.export_dashboard_config(args.export)
        print(f"\n📄 Dashboard config exported to: {args.export}")
    
    print("\n📋 Dashboard Panels:")
    for panel in dashboard.panels:
        print(f"  • {panel.title} ({panel.panel_type.value})")
    
    print("\n🚨 Alert Rules:")
    for alert in dashboard.alert_rules:
        print(f"  • [{alert.severity.value.upper()}] {alert.name}")


def run_all(args):
    """Run all tools and generate comprehensive report."""
    from estimators.flops_estimator import FLOPEstimator, CONFIGS as FLOP_CONFIGS
    from estimators.memory_estimator import MemoryEstimator, get_config as get_mem_config
    from estimators.param_counter import ParamCounter, CONFIGS as PARAM_CONFIGS
    from diagnostics.routing_diagnostics import create_diagnostics
    from dashboards.team7_dashboard import Team7Dashboard
    
    model = args.model
    
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE MoE TOOLING REPORT: {model.upper()}")
    print(f"{'='*60}")
    
    report = {'model': model}
    
    # Parameters
    if model in PARAM_CONFIGS:
        counter = ParamCounter(PARAM_CONFIGS[model])
        report['parameters'] = counter.full_report()
        print(f"\n✓ Parameter count: {report['parameters']['summary']['total_params']}")
    
    # FLOPs
    if model in FLOP_CONFIGS:
        flop_est = FLOPEstimator(FLOP_CONFIGS[model])
        report['flops'] = flop_est.full_report()
        print(f"✓ FLOPs/token: {report['flops']['totals']['total_per_token']}")
    
    # Memory
    model_cfg, dist_cfg = get_mem_config(model)
    mem_est = MemoryEstimator(model_cfg, dist_cfg)
    report['memory'] = mem_est.full_report()
    print(f"✓ Per-GPU Memory: {report['memory']['distributed']['total_per_gpu']}")
    
    # Diagnostics config
    diagnostics = create_diagnostics(model)
    report['diagnostics'] = {
        'num_routed_experts': diagnostics.config.num_routed_experts,
        'num_null_experts': diagnostics.config.num_null_experts,
        'thresholds': {
            'min_null_junk_rate': diagnostics.config.min_null_junk_rate,
            'max_null_signal_rate': diagnostics.config.max_null_signal_rate,
            'min_entropy': diagnostics.config.min_routing_entropy,
        }
    }
    print(f"✓ Diagnostics configured for {diagnostics.config.num_routed_experts} experts")
    
    # Dashboard
    dashboard = Team7Dashboard()
    report['dashboard'] = {
        'panels': len(dashboard.panels),
        'alerts': len(dashboard.alert_rules),
    }
    print(f"✓ Dashboard with {len(dashboard.panels)} panels, {len(dashboard.alert_rules)} alerts")
    
    # Export
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n📄 Full report exported to: {args.output}")
    
    print(f"\n{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='MoE Architecture Tooling Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py estimate 70b_moe
  python cli.py estimate 3b_moe --output report.json
  python cli.py profile --interval 10
  python cli.py diagnose --model 70b_moe --export telemetry.json
  python cli.py dashboard --export dashboard.json
  python cli.py all 70b_moe --output full_report.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Estimate command
    est_parser = subparsers.add_parser('estimate', help='Run estimation tools')
    est_parser.add_argument('model', choices=['3b_moe', '8b_moe', '70b_moe'], help='Model size')
    est_parser.add_argument('--output', '-o', help='Output file for report')
    
    # Profile command
    prof_parser = subparsers.add_parser('profile', help='Configure training profiler')
    prof_parser.add_argument('--interval', type=int, default=10, help='Profile every N steps')
    prof_parser.add_argument('--warmup', type=int, default=5, help='Warmup steps')
    prof_parser.add_argument('--tensorboard', action='store_true', help='Enable TensorBoard')
    
    # Diagnose command
    diag_parser = subparsers.add_parser('diagnose', help='Configure routing diagnostics')
    diag_parser.add_argument('--model', '-m', choices=['3b_moe', '8b_moe', '70b_moe'], default='70b_moe')
    diag_parser.add_argument('--export', '-e', help='Export telemetry config')
    
    # Dashboard command
    dash_parser = subparsers.add_parser('dashboard', help='Generate dashboard configuration')
    dash_parser.add_argument('--export', '-e', help='Export dashboard config')
    
    # All command
    all_parser = subparsers.add_parser('all', help='Run all tools')
    all_parser.add_argument('model', choices=['3b_moe', '8b_moe', '70b_moe'], help='Model size')
    all_parser.add_argument('--output', '-o', help='Output file for full report')
    
    args = parser.parse_args()
    
    if args.command == 'estimate':
        run_estimate(args)
    elif args.command == 'profile':
        run_profile(args)
    elif args.command == 'diagnose':
        run_diagnose(args)
    elif args.command == 'dashboard':
        run_dashboard(args)
    elif args.command == 'all':
        run_all(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
