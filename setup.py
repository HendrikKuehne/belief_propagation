from setuptools import setup,find_packages

setup(
    name="belief_propagation",
    version = "0.0.1",
    description="Algorithms for Belief Propagation on different graphs.",
    author="Hendrik Kühne",
    author_email="hendrik.kuehne2@gmail.com",
    url="https://github.com/HendrikKuehne/belief_propagation.git",
    packages=find_packages(include=['belief_propagation', 'belief_propagation.*']),
    install_requires=[
        "numpy",
        "networkx",
        "matplotlib",
        "cotengra",
        "scipy",
        "sparse"
    ],
)