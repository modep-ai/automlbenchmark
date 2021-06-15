"""
benchmark runners
"""

from .aws import AWSBenchmark
from .docker import DockerBenchmark, DockerBenchmarkAPI
from .singularity import SingularityBenchmark

__all__ = [
    "DockerBenchmark",
    "SingularityBenchmark",
    "AWSBenchmark",
    "DockerBenchmarkAPI",
]
