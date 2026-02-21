import json

import pandas as pd
import streamlit as st
from compute import (
    TrainingStage,
    apply_growth_allocation,
    get_scaling_efficiency,
    get_zero_efficiency,
)

# Page Config
st.set_page_config(layout="wide", page_title="ERA V4 Compute Calculator")

st.title("🧮 ERA V4 Compute & Schedule Calculator")

# Sidebar: Hardware Configuration
st.sidebar.header("Hardware Configuration")

config_path = "experiments/9_training_stack_optimisation_and_cost_governor/FLOPS-Calculation/config.json"


@st.cache_data
def load_config_file():
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load config file: {e}")
        return {}


# Load Initial Config
if "config" not in st.session_state:
    st.session_state.config = load_config_file()

if not st.session_state.config:
    st.stop()

config = st.session_state.config
hw = config.get("hardware", {})

# GPU & Pricing
col1, col2 = st.sidebar.columns(2)
num_gpus = col1.number_input(
    "Num GPUs", min_value=1, value=hw.get("num_gpus", 8), step=1
)
price_per_gpu = col2.number_input(
    "Price/GPU/Hr ($)", min_value=0.0, value=hw.get("price_per_gpu_hour", 2.5), step=0.1
)

# Efficiency
mfu = st.sidebar.slider(
    "Base MFU (Single Node / GEMM)",
    0.1,
    1.0,
    hw.get("mfu", 0.30),
    help="Optimization level of the kernel itself (e.g. FlashAttention-2). Before distributed overheads.",
)

# ZeRO & Offload
zero_stage = st.sidebar.selectbox(
    "ZeRO Stage", [0, 2, 3], index=[0, 2, 3].index(hw.get("zero_stage", 2))
)
cpu_offload = st.sidebar.checkbox(
    "CPU Offload (ZeRO-Infinity)", value=hw.get("cpu_offload", False)
)

# Quantization
quantization_options = ["bf16", "fp8", "nvfp4"]
selected_precisions = st.sidebar.multiselect(
    "Precisions to Calculate",
    quantization_options,
    default=(
        quantization_options
        if hw.get("quantization") == "all"
        else [hw.get("quantization", "bf16")]
    ),
)

# TFLOPS Specs
with st.sidebar.expander("TFLOPS Specs"):
    tflops_bf16 = st.number_input(
        "BF16 TFLOPS", value=hw["tflops_per_gpu"].get("bf16", 989.0)
    )
    tflops_fp8 = st.number_input(
        "FP8 TFLOPS", value=hw["tflops_per_gpu"].get("fp8", 1979.0)
    )
    tflops_nvfp4 = st.number_input(
        "NVFP4 TFLOPS", value=hw["tflops_per_gpu"].get("nvfp4", 3500.0)
    )

    current_tflops_specs = {
        "bf16": tflops_bf16,
        "fp8": tflops_fp8,
        "nvfp4": tflops_nvfp4,
    }

# Advanced Efficiency Config
with st.sidebar.expander("Efficiency Calibration"):
    st.markdown("### ZeRO Efficiency")

    # Defaults
    def_zero_eff = hw.get(
        "zero_efficiency",
        {"zero0": 1.0, "zero2": 0.95, "zero3": 0.70, "zero_infinity": 0.25},
    )

    z0 = st.slider("ZeRO-0 Eff", 0.1, 1.0, def_zero_eff.get("zero0", 1.0))
    z2 = st.slider("ZeRO-2 Eff", 0.1, 1.0, def_zero_eff.get("zero2", 0.95))
    z3 = st.slider("ZeRO-3 Eff", 0.1, 1.0, def_zero_eff.get("zero3", 0.70))
    z_inf = st.slider("ZeRO-Inf Eff", 0.1, 1.0, def_zero_eff.get("zero_infinity", 0.25))

    zero_efficiency_cfg = {
        "zero0": z0,
        "zero2": z2,
        "zero3": z3,
        "zero_infinity": z_inf,
    }

    st.markdown("### Scaling Efficiency")
    def_scale_eff = hw.get(
        "scaling_efficiency",
        {
            "base_gpus": 8,
            "efficiency_at_base": 1.0,
            "efficiency_at_64": 0.90,
            "efficiency_at_256": 0.80,
            "efficiency_at_1024": 0.65,
        },
    )

    eff_64 = st.slider(
        "Eff @ 64 GPUs", 0.1, 1.0, def_scale_eff.get("efficiency_at_64", 0.90)
    )
    eff_256 = st.slider(
        "Eff @ 256 GPUs", 0.1, 1.0, def_scale_eff.get("efficiency_at_256", 0.80)
    )
    eff_1024 = st.slider(
        "Eff @ 1024 GPUs", 0.1, 1.0, def_scale_eff.get("efficiency_at_1024", 0.65)
    )

    scaling_efficiency_cfg = {
        "base_gpus": 8,
        "efficiency_at_base": 1.0,
        "efficiency_at_64": eff_64,
        "efficiency_at_256": eff_256,
        "efficiency_at_1024": eff_1024,
    }

# Main Area: Stage Configuration
st.subheader("Training Stages")

stages_data = []
for stage in config.get("stages", []):
    arch = stage["architecture"]
    stages_data.append(
        {
            "name": stage["name"],
            "total_tokens (B)": stage["total_tokens"] / 1e9,
            "vocab_size": arch.get("vocab_size", 50257),
            "hidden_size": arch.get("hidden_size", 2048),
            "intermediate_size": arch.get("intermediate_size", 0),
            "num_layers": arch.get("num_layers", 24),
            "num_heads": arch.get("num_heads", 16),
            "num_experts": arch.get("num_experts", 0),
            "top_k_experts": arch.get("top_k_experts", 0),
            "null_expert_prob": arch.get("null_expert_prob", 0.0),
            "sequence_length": arch.get("sequence_length", 4096),
            "actual_tokens (B)": arch.get("actual_tokens", stage["total_tokens"]) / 1e9,
        }
    )

edited_stages = st.data_editor(stages_data, num_rows="dynamic")

# Button to Calculate
st.divider()
calculate_btn = st.button("🚀 Calculate Compute & Cost", type="primary")

if calculate_btn:
    # Reconstruct Config
    updated_stages = []
    for s in edited_stages:
        updated_stages.append(
            {
                "name": s["name"],
                "total_tokens": float(s["total_tokens (B)"]) * 1e9,
                "architecture": {
                    "vocab_size": int(s["vocab_size"]),
                    "hidden_size": int(s["hidden_size"]),
                    "intermediate_size": int(s["intermediate_size"]),
                    "num_layers": int(s["num_layers"]),
                    "num_heads": int(s["num_heads"]),
                    "num_experts": int(s["num_experts"]),
                    "top_k_experts": int(s["top_k_experts"]),
                    "null_expert_prob": float(s.get("null_expert_prob", 0.0)),
                    "sequence_length": int(s["sequence_length"]),
                    "tie_embeddings": True,
                    "actual_tokens": float(s["actual_tokens (B)"]) * 1e9,
                },
            }
        )

    # Init TrainingStages
    training_stages = [
        TrainingStage(
            name=s["name"],
            total_tokens=s["total_tokens"],
            architecture=s["architecture"],
            description="",
        )
        for s in updated_stages
    ]

    # Calculate Efficiencies
    zero_eff = get_zero_efficiency(zero_stage, cpu_offload, zero_efficiency_cfg)
    scaling_eff = get_scaling_efficiency(num_gpus, scaling_efficiency_cfg)
    effective_mfu = mfu * zero_eff * scaling_eff

    offload_str = " + CPU Offload" if cpu_offload else ""

    st.markdown("### 📊 Compute Efficiency Breakdown")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Base MFU", f"{mfu*100:.1f}%", help="Raw kernel efficiency (Single GPU)")
    c2.metric(
        "ZeRO Efficiency",
        f"{zero_eff*100:.1f}%",
        f"{(zero_eff-1)*100:.1f}% Penalty",
        delta_color="inverse",
        help=f"Overhead from ZeRO-{zero_stage}{offload_str}",
    )
    c3.metric(
        "Scaling Efficiency",
        f"{scaling_eff*100:.1f}%",
        f"{(scaling_eff-1)*100:.1f}% Penalty",
        delta_color="inverse",
        help=f"Communication overhead for {num_gpus} GPUs",
    )
    c4.metric(
        "Effective MFU",
        f"{effective_mfu*100:.1f}%",
        f"{(effective_mfu-mfu)*100:.1f}% Total Drop",
        delta_color="inverse",
        help="Final efficiency used for calculation",
    )

    st.divider()

    # Apply Growth Allocation
    apply_growth_allocation(training_stages, {"mode": "paper"})

    tabs = st.tabs([p.upper() for p in selected_precisions])

    for i, precision in enumerate(selected_precisions):
        with tabs[i]:
            peak_tflops = current_tflops_specs.get(precision, 0)
            if peak_tflops == 0:
                st.warning(f"No TFLOPS spec for {precision}")
                continue

            effective_flops_per_sec = num_gpus * (peak_tflops * 1e12) * effective_mfu
            effective_pflops = effective_flops_per_sec / 1e15

            st.markdown(
                f"**Effective Cluster Performance:** `{effective_pflops:.2f} PFLOPS`"
            )

            results = []
            total_cost = 0
            total_days = 0
            total_flops = 0

            for stage in training_stages:
                params = stage.calculate_params()
                stage_flops = stage.calculate_flops(params)
                stage_seconds = stage_flops / effective_flops_per_sec
                stage_days = stage_seconds / (24 * 3600)
                stage_cost = (stage_seconds / 3600) * num_gpus * price_per_gpu

                # Memory
                mem_info = stage.calculate_memory_per_gpu(
                    params, num_gpus, zero_stage, precision, cpu_offload
                )
                mem_gb = mem_info["memory_per_gpu_gb"]
                mem_status = "✅"
                if mem_gb > 80:
                    mem_status = "⚠️ >80GB"

                mem_display = f"{mem_gb:.1f} {mem_status}"
                if cpu_offload:
                    cpu_mem_gb = mem_info.get("cpu_memory_per_gpu_gb", 0)
                    mem_display = f"{mem_gb:.1f} GPU | {cpu_mem_gb:.0f} CPU"

                results.append(
                    {
                        "Stage": stage.name,
                        "Tokens (B)": f"{stage.total_tokens/1e9:.1f}",
                        "Total Params (B)": f"{params['total_params']/1e9:.1f}",
                        "Active Params (B)": f"{params['active_params']/1e9:.1f}",
                        "Mem (GB)": mem_display,
                        "ZFLOPs": f"{stage_flops/1e21:.2f}",
                        "Days": f"{stage_days:.2f}",
                        "Cost": f"${stage_cost:,.0f}",
                    }
                )

                total_cost += stage_cost
                total_days += stage_days
                total_flops += stage_flops

            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)

            # Totals
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Cost", f"${total_cost:,.0f}")
            c2.metric("Total Days", f"{total_days:.1f}")
            c3.metric("Total ZFLOPs", f"{total_flops/1e21:.2f}")

# Save Config Button
with st.sidebar:
    st.divider()
    if st.button("💾 Save Config to JSON"):
        # Construct full config dict
        # Note: This is a simplified reconstruction. In a real app we'd bind inputs directly to the state.
        new_config = {
            "hardware": {
                "num_gpus": num_gpus,
                "price_per_gpu_hour": price_per_gpu,
                "mfu": mfu,
                "quantization": "all",
                "zero_stage": zero_stage,
                "cpu_offload": cpu_offload,
                "tflops_per_gpu": current_tflops_specs,
                "zero_efficiency": zero_efficiency_cfg,
                "scaling_efficiency": scaling_efficiency_cfg,
            },
            "growth": {"mode": "paper"},
            "stages": [],
        }

        # We need to capture the edited stages again
        for s in edited_stages:
            new_config["stages"].append(
                {
                    "name": s["name"],
                    "total_tokens": float(s["total_tokens (B)"]) * 1e9,
                    "architecture": {
                        "vocab_size": int(s["vocab_size"]),
                        "hidden_size": int(s["hidden_size"]),
                        "intermediate_size": int(s["intermediate_size"]),
                        "num_layers": int(s["num_layers"]),
                        "num_heads": int(s["num_heads"]),
                        "num_experts": int(s["num_experts"]),
                        "top_k_experts": int(s["top_k_experts"]),
                        "null_expert_prob": float(s.get("null_expert_prob", 0.0)),
                        "sequence_length": int(s["sequence_length"]),
                        "tie_embeddings": True,
                        "actual_tokens": float(s["actual_tokens (B)"]) * 1e9,
                    },
                }
            )

        try:
            with open(config_path, "w") as f:
                json.dump(new_config, f, indent=2)
            st.success("Config saved successfully!")
            st.session_state.config = new_config  # Update session state
        except Exception as e:
            st.error(f"Error saving config: {e}")
