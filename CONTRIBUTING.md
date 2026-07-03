# Contributing to Bensemble

First of all, thank you for your interest in contributing to **Bensemble**!

Whether you found a bug, have an idea for a new Bayesian learning algorithm, want to improve the documentation, or fix a typo, every contribution is appreciated.

## Reporting Issues

If you encounter a bug or unexpected behavior, please open a GitHub Issue and include:

* a clear description of the problem;
* a minimal reproducible example (if possible);
* your Python version;
* your PyTorch version;
* your operating system.

If you have an idea for a new feature or algorithm, feel free to open an Issue first so we can discuss the design before implementation.

---

## Development Setup

We recommend using **uv** for development.

Clone the repository:

```bash
git clone https://github.com/intsystems/bensemble.git
cd bensemble
```

Create a virtual environment:

```bash
uv venv
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

Install the project together with development dependencies:

```bash
uv sync --all-groups
```

---

## Running Tests

Before submitting a Pull Request, please ensure that all tests pass.

```bash
pytest tests/
```

---

## Code Style

We use **Ruff** for linting and formatting.

Before committing, run:

```bash
ruff check .
ruff format .
```

---

## Pull Requests

When submitting a Pull Request, please:

* keep changes focused on a single feature or bug fix;
* add or update tests when introducing new functionality;
* update the documentation if the public API changes;
* ensure that all tests pass;
* write clear and descriptive commit messages.

Small Pull Requests are generally easier to review than very large ones.

---

## Documentation

Documentation is an important part of the project.

If you introduce new functionality, please consider updating:

* the API documentation;
* tutorials or notebooks (when appropriate);
* the README if the feature is user-facing.

---

## Coding Guidelines

Bensemble aims to provide clean, modular implementations of Bayesian Deep Learning and Neural Network Ensembling methods.

When contributing new algorithms, please try to:

* follow the existing project structure;
* reuse existing abstractions where possible;
* keep implementations readable and well documented;
* avoid introducing unnecessary dependencies.

---

## Questions

If you are unsure about an implementation or would like feedback before starting larger work, feel free to open a GitHub Issue or Discussion.

We appreciate every contribution that helps make Bensemble better.
