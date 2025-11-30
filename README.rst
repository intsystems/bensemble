|test| |codecov| |docs|

.. |test| image:: https://github.com/intsystems/ProjectTemplate/workflows/test/badge.svg
    :target: https://github.com/intsystems/ProjectTemplate/tree/master
    :alt: Test status
    
.. |codecov| image:: https://img.shields.io/codecov/c/github/intsystems/ProjectTemplate/master
    :target: https://app.codecov.io/gh/intsystems/ProjectTemplate
    :alt: Test coverage
    
.. |docs| image:: https://github.com/intsystems/ProjectTemplate/workflows/docs/badge.svg
    :target: https://intsystems.github.io/ProjectTemplate/
    :alt: Docs status


.. class:: center

    :Название исследуемой задачи: Bayesian ensembling
    :Тип научной работы: Учебный проект по курсу байесовского мультимоделирования
    :Автор: Соболевский Федор, Набиев Мухаммадшариф, Василенко Дмитрий, Касюк Вадим
    :Научный руководитель: к. ф.-м. н. Бахтеев Олег Юрьевич
    :Научный консультант(при наличии): степень, Фамилия Имя Отчество

Abstract: Comparative Analysis of Bayesian Deep Learning Methods for Multimodel Ensembling
========

Bayesian methods provide a principled framework for uncertainty quantification in deep learning, which is crucial for safety-critical applications. This project presents a comprehensive comparative study of four prominent variational inference algorithms for Bayesian neural networks: the baseline Evidence Lower Bound (ELBO) with local reparameterization trick, hyperparameter optimization, and pruning; Renyi divergence as a generalization of ELBO offering tunable mode-seeking/covering behavior; scalable Laplace approximation for efficient posterior estimation; and Bayes by Backpropagation as an alternative variational inference scheme.

We implement these methods in a unified framework and evaluate them on benchmark classification and regression tasks using metrics including predictive accuracy, negative log-likelihood, expected calibration error, and out-of-distribution detection performance. Our results demonstrate the trade-offs between computational efficiency, uncertainty quantification quality, and model performance across different methods. Specifically, we show how Renyi divergence with α < 1 provides improved uncertainty calibration compared to standard ELBO, while scalable Laplace approximation offers competitive performance with reduced computational overhead. The study provides practical guidance for selecting appropriate Bayesian inference methods based on application requirements and computational constraints.

Keywords: Bayesian Deep Learning, Variational Inference, Uncertainty Quantification, Renyi Divergence, Laplace Approximation, Neural Networks Ensembling

Bayesian Multimodeling Project
==============================

A comprehensive comparative study of Bayesian deep learning methods for neural network ensembling and uncertainty quantification.

Project documentation can be found here: https://intsystems.github.io/bensemble/

Blog post draft: https://github.com/intsystems/bensemble/blob/master/paper/blogpost_draft/blog_post_draft.pdf

Techreport draft: https://github.com/intsystems/bensemble/blob/master/paper/techreport_draft/bensemble_Tech_Report_draft.pdf

Benchmark notebook: https://github.com/intsystems/bensemble/blob/master/notebooks/benchmark.ipynb

Demos
--------------------
Practical Variational Inference: https://github.com/intsystems/bensemble/blob/master/notebooks/pvi_demo.ipynb

Kronecker-Factored Laplace Approximation: https://github.com/intsystems/bensemble/blob/master/notebooks/laplace_demo.ipynb

Variational Inference with Renyi Divergence: https://github.com/intsystems/bensemble/blob/master/notebooks/variatinal_renyi_demo.ipynb

Bayes by Backprop: https://github.com/intsystems/bensemble/blob/master/notebooks/pbp_probabilistic_backpropagation_test.ipynb 
