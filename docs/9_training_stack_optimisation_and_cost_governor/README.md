# Team 9: Training Stack Optimization & Cost Governor

Owner: Infrastructure / Training Systems

This folder tracks Team 9 charter deliverables and links to the concrete artifacts in `experiments/9_training_stack_optimisation_and_cost_governor/deliverables/`.

## Scope
- Training stack selection and configuration
- Sharding strategy and activation checkpointing
- Data-loader and pipeline throughput instrumentation
- Runtime profiling and cost-per-token accounting
- Automatic HALT + safe shutdown controller

## Deliverables
- Training configs (primary + fallback per stage)
- Cost-per-token tables and budget envelopes
- Throughput and waste attribution report
- Automatic HALT system (scripts + triggers)
- Failure and recovery notes
- NVFP4 readiness notes

## Artifact Index
- Training configs: `experiments/9_training_stack_optimisation_and_cost_governor/deliverables/training_configs.md`
- Instrumentation setup: `experiments/9_training_stack_optimisation_and_cost_governor/deliverables/instrumentation_setup.md`
- Budget modeling: `experiments/9_training_stack_optimisation_and_cost_governor/deliverables/budget_modeling.md`
- Cost-per-token table: `experiments/9_training_stack_optimisation_and_cost_governor/deliverables/cost_per_token_tables.csv`
- Throughput & waste report: `experiments/9_training_stack_optimisation_and_cost_governor/deliverables/throughput_waste_report.md`
- HALT system: `experiments/9_training_stack_optimisation_and_cost_governor/deliverables/halt_system/README.md`
- Failure & recovery notes: `experiments/9_training_stack_optimisation_and_cost_governor/deliverables/failure_recovery_notes.md`
- NVFP4 readiness: `experiments/9_training_stack_optimisation_and_cost_governor/deliverables/nvfp4_readiness.md`
