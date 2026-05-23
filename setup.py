from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="UniDISA",
    version="0.1.0",
    author="Zhengfang Lu",
    description="Unified framework for Diagonal Integration via Stagewise Alignment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/whitezb12/UniDISA",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0,<3.0",
        "scanpy>=1.9.0,<2.0",
        "scib",
        "numpy",
        "pandas",
        "anndata",
        "scib_metrics",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)