"""Keep the installed runtime's existing skill/algorithm/cache layout."""

from setuptools import find_packages, setup

packages = [
    "scheduleurm",
    "scheduleurm.skill",
    "scheduleurm.skill.integrations",
    "scheduleurm.md.experiment_artifacts",
    "scheduleurm.algorithm",
    "scheduleurm.simulation",
]
packages += [
    f"scheduleurm.skill.{name}"
    for name in find_packages("skill", include=["scheduler_*", "scheduler_*.*"])
]
packages += [f"scheduleurm.algorithm.{name}" for name in find_packages("algorithm")]

setup(
    packages=packages,
    package_dir={
        "scheduleurm": "packaging/scheduleurm",
        "scheduleurm.skill": "skill",
        "scheduleurm.algorithm": "algorithm",
        "scheduleurm.simulation": "simulation",
        "scheduleurm.md.experiment_artifacts": "md/experiment_artifacts",
    },
)
