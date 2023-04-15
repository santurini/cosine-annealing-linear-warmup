from setuptools import setup

setup(
    name="cosine_warmup",
    version="0.1",
    author="Arturo Ghinassi",
    packages=['cosine_warmup'],
    description="Cosine Annealing Scheduler with Linear Warmup and multiple groups support for PyTorch",
    long_description=open("README.md").read(),
)
