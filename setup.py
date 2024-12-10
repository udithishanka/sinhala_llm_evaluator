from setuptools import setup, find_packages

setup(
    name="sinhala_llm_evaluator",
    version="0.2.0",  
    description="A module for evaluating language models during training",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/udithishanka",  # Your repo or homepage
    author="Udith Ishanka",
    author_email="udithishanka.s@gmail.com",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "datasets",
        "evaluate",
        "numpy",
        "nltk",
        "absl-py",
        "rouge-score",
        "scikit-learn"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
