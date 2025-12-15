# Welcome to Bensemble

*Pro tip: visit our project's [GitHub page](https://github.com/intsystems/bensemble).*

## About Bensemble

***Bensemble*** is a lightweight library based on PyTorch that implements several Bayesian (probabilistic) ensembling methods for linear neural network models. Bayesian ensembling is a method of making predictions in machine learning that incorporate uncertainty about the predictions into the estimate. This is done by approximating the posterior distribution over the weights of the neural network and then using the distribution to either sample predictions or *ensembles* of models, the predictions of which are then averaged and analyzed for uncertainty estimation.

## Navigation

To get a quickstart on library installation and basic usage, go to the [Getting Started](getting-started) page.

To get a more detailed insight into how each method works, check out the [User Guide](user-guide/interface) section. There you can find guides for each implemented method as well as a description of the general interface of our ensemble classes (go to [Interface](user-guide/interface) for more).

For specific details on the API, go to the [API](api/methods) section.

We also provide links to the papers describing all the implemented methods in the [References](references) section.