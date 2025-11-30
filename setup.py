from setuptools import setup, find_packages

install_requires = [
    "torch==2.9.1",
    "numpy==2.2.6",
    "scikit-learn==1.7.2",
    "matplotlib==3.10.7",
    "scipy==1.15.3"
]

extras_require = {
    "dev": [
        "black",
        "isort",
        "nbqa",
        "pytest",
        "mypy",
        "autopep8"
    ]
}

setup(
    name="bensemble",  
    version="0.1.0",
    packages=find_packages(), 
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires=">=3.10",
    description="Bayesian ensemble methods for neural networks",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/intsystems/bensemble",
    author="Соболевский Федор, Набиев Мухаммадшариф, Василенко Дмитрий, Касюк Вадим",
    author_email="kasukvadim@mail.ru",
    license="MIT",
    keywords="bayesian neural network ensemble variational inference renyi",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    project_urls={
        "Bug Tracker": "https://github.com/intsystems/bensemble/issues",
        "Documentation": "https://github.com/intsystems/bensemble#readme",
        "Source": "https://github.com/intsystems/bensemble",
    },
    include_package_data=True, 
)
