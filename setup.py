"""
MAL-ICS++ Package Setup
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="malics-plusplus",
    version="1.0.0",
    author="MAL-ICS++ Research Team",
    author_email="your-email@institution.edu",
    description="Multi-Layer Malware-Resilient FDIA Detection Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/malics-plusplus",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandapower>=3.2.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.6.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
        "joblib>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.6b0",
            "flake8>=3.9.0",
            "ipython>=7.25.0",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "malics-train=models.train_classifiers:main",
            "malics-generate=dataset.generate_master_dataset:main",
        ],
    },
)
