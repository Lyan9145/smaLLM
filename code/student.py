"""Submission factory; include student_model.py with every inference bundle."""
from student_model import StudentLM


def build_model(config):
    return StudentLM(config)
