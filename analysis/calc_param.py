from __future__ import annotations

from analysis.pipeline import AnalysisResult


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    return total, trainable, non_trainable


def analyze_params(_args, model, **_kwargs) -> list[AnalysisResult]:
    """
    Pipeline-compatible parameter counter.

    Returns a single .txt AnalysisResult with total / trainable / non-trainable counts.
    Saved as:  analysis_output/params/{project}/{model_name}_params.txt
    """
    total, trainable, non_trainable = count_parameters(model)

    lines = [
        f"total        : {total:,}",
        f"trainable    : {trainable:,}",
        f"non-trainable: {non_trainable:,}",
    ]
    text = "\n".join(lines) + "\n"

    print(f"[params] total={total:,}  trainable={trainable:,}  non-trainable={non_trainable:,}")

    return [AnalysisResult(suffix="params", data=text, fmt=".txt")]
