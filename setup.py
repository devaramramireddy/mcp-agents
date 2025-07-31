from setuptools import setup, find_packages

setup(
    name="mcpuse",
    version="0.1.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "pyyaml"
    ],
    extras_require={"vllm": ["vllm"]},
)
