"""Evaluator-facing factory for the optional top-1 MoE student."""

from moe_model import MoEStudentLM


def build_model(config):
    return MoEStudentLM(config)
