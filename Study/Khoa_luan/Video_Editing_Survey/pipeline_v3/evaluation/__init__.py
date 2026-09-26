"""
Evaluation suite for Pipeline v3:
- eval_stage1_geometry: Evaluates DoRA Geometry Adaptation against Baseline & Scaffold
- eval_stage2_appearance: Evaluates Ref-DiT Perceiver Cross-Attention against Stage 1
- eval_final_ablation: Complete multi-column visual benchmark across all stages
- eval_checkpoint_evolution: SOTA Baseline vs Step 0 vs All Checkpoint Versions
- show_comparator: Quick side-by-side web video comparator for Jupyter Notebook
"""

from .show_comparator import show

__all__ = ["show"]
