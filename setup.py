from setuptools import setup, find_packages

setup(
    name="hbsattn",
    version="0.0.1",
    description="Block-sparse attention with heterogenoues block size",
    author="Zhengqi Gao",
    author_email="zhengqi@mit.edu",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "triton",
    ],
    extras_require={
        "dev": ["pytest"],
    },
    python_requires=">=3.8",
    include_package_data=True,
    zip_safe=False,
)
