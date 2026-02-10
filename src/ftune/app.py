"""
ftune Web UI — Interactive LLM Fine-Tuning Cost Calculator

Run with:
    streamlit run app.py

Or from the project root:
    streamlit run src/ftune/app.py
"""

import sys
from pathlib import Path

# Ensure ftune is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from ftune import Estimator, list_model_names, list_gpu_names
from ftune.core.models import FineTuneMethod, Quantization, LoRATarget, OptimizerType
from ftune.loader import load_gpus


# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="ftune — LLM Fine-Tuning Calculator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .main-header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.2rem;
    }
    .main-header p {
        color: #888;
        font-size: 1.1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #333;
    }
    .metric-card h3 {
        color: #a0a0a0;
        font-size: 0.85rem;
        margin-bottom: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
    }
    .fit-yes { color: #4ade80; }
    .fit-no { color: #f87171; }
    .stTabs [data-baseweb="tab-list"] { gap: 2rem; }
    div[data-testid="stMetricValue"] { font-size: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>⚡ ftune</h1>
    <p>Estimate GPU memory, training time, and cloud costs for LLM fine-tuning</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────
# Sidebar — Configuration
# ─────────────────────────────────────────────

with st.sidebar:
    st.header("🔧 Configuration")

    # Model selection
    st.subheader("Model")
    model_options = list_model_names()
    use_custom = st.checkbox("Use custom HuggingFace model ID", value=False)

    if use_custom:
        model_name = st.text_input(
            "HuggingFace Model ID",
            value="meta-llama/Llama-3.1-8B",
            help="Any model on HuggingFace Hub (e.g. 'mistralai/Mistral-7B-v0.3')",
        )
    else:
        model_name = st.selectbox("Select Model", model_options, index=model_options.index("meta-llama/Llama-3.1-8B"))

    # Fine-tuning method
    st.subheader("Training Method")
    method = st.selectbox(
        "Method",
        ["qlora", "lora", "full"],
        format_func=lambda x: {"qlora": "QLoRA (4-bit quantized)", "lora": "LoRA", "full": "Full Fine-Tuning"}[x],
    )

    # Method-specific settings
    if method in ("qlora", "lora"):
        col1, col2 = st.columns(2)
        with col1:
            lora_rank = st.select_slider("LoRA Rank", options=[4, 8, 16, 32, 64, 128, 256], value=16)
        with col2:
            lora_alpha = st.select_slider("LoRA Alpha", options=[8, 16, 32, 64, 128], value=32)

        lora_target = st.selectbox(
            "Target Modules",
            ["attention", "attention_all", "all_linear"],
            format_func=lambda x: {
                "attention": "Q, V projections only",
                "attention_all": "All attention (Q, K, V, O)",
                "all_linear": "All linear layers (attention + MLP)",
            }[x],
        )
    else:
        lora_rank, lora_alpha, lora_target = 16, 32, "attention"

    quantization = "4bit" if method == "qlora" else ("8bit" if method == "qlora" else "none")
    if method == "qlora":
        quantization = st.radio("Quantization", ["4bit", "8bit"], horizontal=True)
    elif method == "lora":
        quantization = "none"
    else:
        quantization = "none"

    # Training settings
    st.subheader("Training Settings")
    col1, col2 = st.columns(2)
    with col1:
        batch_size = st.number_input("Batch Size", min_value=1, max_value=256, value=4)
    with col2:
        seq_length = st.select_slider(
            "Seq Length",
            options=[256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
            value=2048,
        )

    gradient_checkpointing = st.checkbox("Gradient Checkpointing", value=True)

    optimizer = st.selectbox(
        "Optimizer",
        ["adamw", "adam", "sgd", "adam_8bit", "adafactor"],
        format_func=lambda x: {
            "adamw": "AdamW (default)",
            "adam": "Adam",
            "sgd": "SGD",
            "adam_8bit": "8-bit Adam (bitsandbytes)",
            "adafactor": "Adafactor",
        }[x],
    )

    # Dataset settings
    st.subheader("Dataset")
    dataset_size = st.number_input("Number of Samples", min_value=100, max_value=10_000_000, value=50000, step=1000)
    epochs = st.slider("Epochs", min_value=1, max_value=20, value=3)
    num_gpus = st.slider("Number of GPUs", min_value=1, max_value=8, value=1)


# ─────────────────────────────────────────────
# Create Estimator
# ─────────────────────────────────────────────

try:
    est = Estimator(
        model=model_name,
        method=method,
        quantization=quantization,
        batch_size=batch_size,
        seq_length=seq_length,
        gradient_checkpointing=gradient_checkpointing,
        optimizer=optimizer,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_target=lora_target,
    )
    error = None
except Exception as e:
    est = None
    error = str(e)

if error:
    st.error(f"⚠️ Configuration Error: {error}")
    st.stop()


# ─────────────────────────────────────────────
# Compute Estimates
# ─────────────────────────────────────────────

mem = est.estimate_memory()
fits = est.check_gpu_fit()
times = est.estimate_time_all_gpus(dataset_size=dataset_size, epochs=epochs, num_gpus=num_gpus)
full_costs = est.full_comparison(dataset_size=dataset_size, epochs=epochs, num_gpus=num_gpus)


# ─────────────────────────────────────────────
# Top Summary Metrics
# ─────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("💾 Total VRAM", f"{mem.total_gb:.1f} GB")
with col2:
    compatible = [g for g in fits if g.fits]
    st.metric("🖥️ Compatible GPUs", f"{len(compatible)}/{len(fits)}")
with col3:
    fastest = times[0] if times else None
    st.metric("⏱️ Fastest Time", f"{fastest.total_hours:.1f}h" if fastest else "N/A",
              delta=f"on {fastest.gpu_name}" if fastest else None, delta_color="off")
with col4:
    cheapest = full_costs.estimates[0] if full_costs.estimates else None
    st.metric("💰 Cheapest", f"${cheapest.total_cost:.2f}" if cheapest else "N/A",
              delta=f"{cheapest.provider}" if cheapest else None, delta_color="off")


# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(["📊 Memory", "⏱️ Training Time", "💰 Cost Comparison", "🖥️ GPU Compatibility"])


# ── Tab 1: Memory ──
with tab1:
    st.subheader("VRAM Breakdown")

    col1, col2 = st.columns([3, 2])

    with col1:
        # Breakdown table
        components = {
            "Base Model Weights": mem.model_weights_gb,
            "LoRA Adapters": mem.trainable_params_gb,
            "Gradients": mem.gradients_gb,
            "Optimizer States": mem.optimizer_states_gb,
            "Activations": mem.activations_gb,
            "CUDA Overhead": mem.overhead_gb,
        }

        import pandas as pd
        df = pd.DataFrame([
            {"Component": k, "VRAM (GB)": round(v, 3), "% of Total": f"{(v/mem.total_gb)*100:.1f}%"}
            for k, v in components.items()
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown(f"**Total: {mem.total_gb:.2f} GB**")

    with col2:
        # Bar chart
        import altair as alt

        chart_data = pd.DataFrame({
            "Component": list(components.keys()),
            "GB": list(components.values()),
        })
        chart = alt.Chart(chart_data).mark_bar(cornerRadius=4).encode(
            x=alt.X("GB:Q", title="VRAM (GB)"),
            y=alt.Y("Component:N", sort="-x", title=""),
            color=alt.Color("Component:N", legend=None,
                            scale=alt.Scale(scheme="viridis")),
            tooltip=["Component", alt.Tooltip("GB:Q", format=".3f")],
        ).properties(height=250)
        st.altair_chart(chart, use_container_width=True)

    # Trainable params info
    from ftune.utils.formatting import format_params
    st.info(
        f"📐 **Trainable Parameters:** {format_params(mem.trainable_params)} "
        f"({mem.trainable_percentage:.3f}% of {format_params(mem.total_params)} total)"
    )


# ── Tab 2: Training Time ──
with tab2:
    st.subheader(f"Training Time — {dataset_size:,} samples × {epochs} epochs")

    if times:
        time_data = pd.DataFrame([
            {
                "GPU": t.gpu_name,
                "Hours/Epoch": round(t.hours_per_epoch, 1),
                "Total Hours": round(t.total_hours, 1),
                "Total Steps": f"{t.total_steps:,}",
                "Tokens Processed": f"{t.total_tokens:,}",
            }
            for t in times
        ])
        st.dataframe(time_data, use_container_width=True, hide_index=True)

        # Bar chart
        chart_time = alt.Chart(
            pd.DataFrame({"GPU": [t.gpu_name for t in times], "Hours": [t.total_hours for t in times]})
        ).mark_bar(cornerRadius=4).encode(
            x=alt.X("Hours:Q", title="Total Training Hours"),
            y=alt.Y("GPU:N", sort="x", title=""),
            color=alt.Color("Hours:Q", scale=alt.Scale(scheme="redyellowgreen", reverse=True), legend=None),
            tooltip=["GPU", alt.Tooltip("Hours:Q", format=".1f")],
        ).properties(height=max(200, len(times) * 35))
        st.altair_chart(chart_time, use_container_width=True)
    else:
        st.warning("No compatible GPUs found for this configuration.")


# ── Tab 3: Cost Comparison ──
with tab3:
    st.subheader("Cost Across All Providers")

    if full_costs.estimates:
        cost_data = pd.DataFrame([
            {
                "Provider": e.provider,
                "GPU": e.gpu,
                "Training Hours": round(e.training_hours, 1),
                "$/hr (on-demand)": f"${e.hourly_rate:.2f}",
                "Total Cost": round(e.total_cost, 2),
                "Spot $/hr": f"${e.spot_hourly_rate:.2f}" if e.spot_hourly_rate else "—",
                "Spot Total": f"${e.spot_total_cost:.2f}" if e.spot_total_cost else "—",
            }
            for e in full_costs.estimates
        ])
        st.dataframe(cost_data, use_container_width=True, hide_index=True)

        # Top 10 cheapest bar chart
        top10 = full_costs.estimates[:10]
        chart_cost = alt.Chart(
            pd.DataFrame({
                "Option": [f"{e.provider}\n({e.gpu})" for e in top10],
                "Cost ($)": [e.total_cost for e in top10],
            })
        ).mark_bar(cornerRadius=4).encode(
            x=alt.X("Cost ($):Q"),
            y=alt.Y("Option:N", sort="x", title=""),
            color=alt.Color("Cost ($):Q", scale=alt.Scale(scheme="redyellowgreen", reverse=True), legend=None),
            tooltip=["Option", alt.Tooltip("Cost ($):Q", format="$.2f")],
        ).properties(height=max(250, len(top10) * 35))
        st.altair_chart(chart_cost, use_container_width=True)

        # Recommendations
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"💡 **Cheapest on-demand:** {full_costs.cheapest}")
        with col2:
            st.success(f"🏆 **Best value (incl. spot):** {full_costs.best_value}")
    else:
        st.warning("No pricing data available for compatible GPUs.")


# ── Tab 4: GPU Compatibility ──
with tab4:
    st.subheader("GPU Compatibility Check")

    gpu_data = pd.DataFrame([
        {
            "GPU": g.gpu_name,
            "VRAM": f"{g.vram_gb:.0f} GB",
            "Required": f"{g.required_gb:.1f} GB",
            "Fits": "✅ Yes" if g.fits else "❌ No",
            "Utilization": f"{g.utilization_percent:.0f}%",
            "Headroom": f"{g.headroom_gb:.1f} GB" if g.fits else f"{g.headroom_gb:.1f} GB",
        }
        for g in fits
    ])
    st.dataframe(gpu_data, use_container_width=True, hide_index=True)

    # Visual utilization chart
    util_data = pd.DataFrame({
        "GPU": [g.gpu_name for g in fits],
        "Utilization %": [min(g.utilization_percent, 150) for g in fits],
        "Fits": ["Compatible" if g.fits else "Insufficient VRAM" for g in fits],
    })
    chart_util = alt.Chart(util_data).mark_bar(cornerRadius=4).encode(
        x=alt.X("Utilization %:Q", title="VRAM Utilization %",
                 scale=alt.Scale(domain=[0, 150])),
        y=alt.Y("GPU:N", sort="-x", title=""),
        color=alt.Color("Fits:N", scale=alt.Scale(
            domain=["Compatible", "Insufficient VRAM"],
            range=["#4ade80", "#f87171"],
        )),
        tooltip=["GPU", alt.Tooltip("Utilization %:Q", format=".1f"), "Fits"],
    ).properties(height=max(250, len(fits) * 30))

    # Add a 100% reference line
    rule = alt.Chart(pd.DataFrame({"x": [100]})).mark_rule(
        strokeDash=[4, 4], color="#888"
    ).encode(x="x:Q")

    st.altair_chart(chart_util + rule, use_container_width=True)


# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────

st.divider()
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "Built with <a href='https://github.com/ritikmahy5/ftune'>ftune</a> ⚡ "
    "| Estimates are approximate — always validate with a small test run"
    "</div>",
    unsafe_allow_html=True,
)
