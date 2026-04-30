# Neural Ensemble Search

Neural Ensemble Search employs Neural Architecture Search to find an optimal ensemble of models with varying architectures.

To find the optimal ensemble, NES solves a bilevel optimization problem. There are two search strategies described in the original paper:

- **NES-RS (Random Search)**: A simple yet effective strategy relying on random uniform sampling from the search space.
- **NES-RE (Regularized Evolution)**: Utilizes an evolutionary algorithm with tournament selection and mutations to evolve a population of strong and diverse architectures.

---

Sheheryar Zaidi et al.[*"Neural Ensemble Search for Uncertainty Estimation and Dataset Shift"*](https://proceedings.neurips.cc/paper_files/paper/2021/file/41a6fd31aa2e75c3c6d427db3d17ea80-Paper.pdf) (2021)
