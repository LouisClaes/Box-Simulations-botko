"""Setup configuration for Box-Simulations-botko."""
from setuptools import setup, find_packages

setup(
    name="box-simulations-botko",
    version="0.1.0",
    description="3D bin packing framework with stability constraints",
    author="Louis",
    author_email="",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "httpx>=0.27.0",
        "pydantic>=2.10.0",
        "pyyaml>=6.0.0",
        "numpy>=2.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=9.0.0",
            "pytest-asyncio>=0.25.0",
            "pytest-cov>=6.0.0",
            "ruff>=0.15.0",
            "mypy>=1.19.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "botko-run=src.runner.experiment:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
