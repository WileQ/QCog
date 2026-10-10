"""Interactive explorer for Q-Cog (Streamlit).

Run it with::

    streamlit run app/streamlit_app.py

The app is generated from the model metadata and parameter schemas, so every model
in the library - including ones added later - gets an interface with no extra code.

Accessibility is part of the design, not an afterthought:

* every figure is accompanied by its plain-language description and by the data
  table it was drawn from, so the numbers are always reachable without vision;
* a high-contrast option and a larger type size are one checkbox away;
* no interaction depends on hovering or on colour alone;
* all controls are standard Streamlit widgets, which are keyboard operable and
  expose labels to assistive technology;
* a "text only" mode removes the images entirely for users who prefer text.

The app imports the same objects the CLI and reports use, so there is no separate
implementation to keep in sync.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `streamlit run app/streamlit_app.py` from a checkout without installing.
ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "src").exists() and str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import streamlit as st  # noqa: E402

from qcog.core.params import coerce_params  # noqa: E402
from qcog.explain import explain_backend, explain_interference, explain_order_effect, explain_qq  # noqa: E402
from qcog.registry import MODELS, MODULE_ORDER  # noqa: E402
from qcog.report import model_html, write_model_report  # noqa: E402
from qcog.viz import text_alternatives  # noqa: E402

__all__ = ["widget_value", "render_parameters", "main"]


def widget_value(spec, key_prefix: str):
    """Render one parameter widget from its :class:`~qcog.core.params.ParamSpec`.

    Kept as a separate function so the mapping from parameter specification to
    widget can be unit-tested without a running Streamlit server.
    """
    label = spec.display + (f" ({spec.unit})" if spec.unit else "")
    key = f"{key_prefix}:{spec.name}"
    if spec.kind == "float":
        low = 0.0 if spec.low is None else spec.low
        high = 1.0 if spec.high is None else spec.high
        return st.slider(label, min_value=float(low), max_value=float(high),
                         value=float(spec.default), step=float(spec.step or (high - low) / 100),
                         help=spec.description, key=key)
    if spec.kind == "int":
        low = 1 if spec.low is None else int(spec.low)
        high = 100 if spec.high is None else int(spec.high)
        return st.slider(label, min_value=low, max_value=high, value=int(spec.default),
                         help=spec.description, key=key)
    if spec.kind == "bool":
        return st.checkbox(label, value=bool(spec.default), help=spec.description, key=key)
    if spec.kind == "choice":
        options = list(spec.choices or ())
        index = options.index(spec.default) if spec.default in options else 0
        return st.selectbox(label, options, index=index, help=spec.description, key=key)
    return st.text_input(label, value=str(spec.default), help=spec.description, key=key)


def render_parameters(model_class, key_prefix: str = "params") -> dict:
    """Render every parameter of a model and return the chosen values."""
    values: dict[str, object] = {}
    for spec in model_class.params:
        values[spec.name] = widget_value(spec, key_prefix)
    return coerce_params(model_class.params, values)


def _apply_accessibility(dark: bool, large_type: bool, high_contrast: bool) -> None:
    """Inject a small, dependency-free accessibility stylesheet."""
    rules = []
    if dark:
        rules.append("body { background: #101418; color: #f2f2f2; }")
    if large_type:
        rules.append("html { font-size: 118%; }")
    if high_contrast:
        rules.append(
            "body { background: #000 !important; color: #fff !important; }"
            ".stMarkdown, .stCaption, label { color: #fff !important; }"
            "a { color: #ffd166 !important; text-decoration: underline; }"
        )
    if rules:
        st.markdown(f"<style>{' '.join(rules)}</style>", unsafe_allow_html=True)


def main() -> None:
    """Render the application."""
    st.set_page_config(page_title="Q-Cog explorer", page_icon="🧠", layout="wide")
    st.markdown(
        "<style>a:focus-visible, button:focus-visible, summary:focus-visible,"
        "input:focus-visible { outline: 3px solid #d55e00; outline-offset: 2px; }</style>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.title("Q-Cog")
        st.caption("Quantum-probability models of cognition, decision making and groups. "
                   "Every model is simulated with Qiskit and compared with a classical baseline.")
        module = st.selectbox(
            "Module", ["All", "A", "B", "C", "D"],
            format_func=lambda value: value if value == "All" else f"{value}: {MODULE_ORDER[value]}",
        )
        options = [slug for slug, model in MODELS.items() if module in {"All", model.module}]
        slug = st.selectbox(
            "Model", options, format_func=lambda value: MODELS[value].title
        )
        st.subheader("Execution")
        backend = st.selectbox("Backend", ["exact", "sampled", "aer", "aer_exact"])
        shots = st.number_input("Shots", min_value=100, max_value=200000, value=4096, step=100)
        seed = st.number_input("Seed", min_value=0, max_value=10**6, value=7)
        st.subheader("Accessibility")
        dark = st.checkbox("Dark background", value=False)
        large_type = st.checkbox("Larger type", value=False)
        high_contrast = st.checkbox("High contrast", value=False)
        text_only = st.checkbox("Text only (hide charts)", value=False)
        show_tables = st.checkbox("Show the data behind each chart", value=True)

    _apply_accessibility(dark, large_type, high_contrast)

    model_class = MODELS[slug]
    st.header(model_class.title)
    st.caption(f"Module {model_class.module}: {model_class.module_title()}")

    with st.expander("What this model asks, and how it works", expanded=True):
        st.markdown(f"**Question.** {model_class.cognitive_question}")
        st.markdown(f"**Quantum mechanism.** {model_class.quantum_mechanism}")
        st.markdown(f"**Classical baseline.** {model_class.classical_baseline}")
        if model_class.keywords:
            st.markdown("**Keywords.** " + ", ".join(model_class.keywords))
        if model_class.references:
            st.markdown("**References**")
            for reference in model_class.references:
                st.markdown(f"- {reference}")

    st.subheader("Parameters")
    parameters = render_parameters(model_class)
    run = st.button("Run the model", type="primary")

    if not run and f"result:{slug}" not in st.session_state:
        st.info("Set the parameters and activate **Run the model**. "
                "All results are also available from the command line: "
                f"`qcog run {slug} --backend {backend}`.")
        return

    key = f"result:{slug}"
    if run or key not in st.session_state:
        with st.spinner("Simulating with Qiskit..."):
            st.session_state[key] = model_class().run(
                backend=backend, shots=shots, seed=seed, **parameters
            )
    result = st.session_state[key]

    st.subheader("Result")
    st.markdown(f"**{result.summary}**")
    st.caption(explain_backend(next(iter(result.sequential_results().values()))))
    st.markdown("#### What this run shows")
    for finding in result.findings:
        st.markdown(f"- {finding}")

    analysis = result.analysis
    for name, report in (
        ("Order effect", analysis.get("order_effect")),
        ("Interference", analysis.get("interference")),
        ("QQ equality", analysis.get("qq_equality")),
    ):
        if report is None:
            continue
        with st.expander(name):
            if name == "Order effect":
                st.write(explain_order_effect(report))
            elif name == "Interference":
                st.write(explain_interference(report))
            else:
                st.write(explain_qq(report))

    if not text_only:
        st.markdown("#### Charts")
        try:
            figures = model_class().figures(result)
        except Exception as error:  # pragma: no cover - never break the UI on a plot
            figures = {}
            st.warning(f"Charts unavailable: {error}")
        descriptions = text_alternatives(figures)
        for name, figure in figures.items():
            st.markdown(f"**{name}**")
            st.pyplot(figure)
            caption = descriptions.get(name, "")
            if caption:
                st.caption(caption)
                with st.expander("Long description of this chart"):
                    st.write(caption)
            if show_tables:
                for table_name, frame in result.tables().items():
                    if table_name.split("_")[-1] in name or name.split(":")[-1].strip() in table_name:
                        st.dataframe(frame, use_container_width=True)
                        break
    else:
        st.info("Charts are hidden. The tables below contain the same numbers.")
        for table_name, frame in result.tables().items():
            st.markdown(f"**{table_name.replace('_', ' ')}**")
            st.dataframe(frame, use_container_width=True)

    st.markdown("#### Limit what this does not claim")
    for caveat in result.caveats:
        st.markdown(f"- {caveat}")

    st.divider()
    st.markdown("#### Export")
    st.download_button(
        "Download the accessible HTML report for this run",
        data=model_html(model_class, result, include_figures=True).encode("utf-8"),
        file_name=f"{slug}.html",
        mime="text/html",
        help="Self-contained HTML: no JavaScript, no network, with chart descriptions and data tables.",
    )
    st.caption("This effort is supported by Qollab & IonQ.")


if __name__ == "__main__":  # pragma: no cover - Streamlit executes the module
    main()