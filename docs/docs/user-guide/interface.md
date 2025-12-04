# Interface

The interface of all of the models implemented in Bensemble is (almost) the same to provide interchangeability. In this section we will explore the common methods used in all of Bensemble's classes.

## Initialization (`__init__`)

All of the methods' constructors take different input arguments, however, one thing that most of them have in common is that they require a `nn.Module` model as their first argument. This is because Bensemble models are in fact wrappers around PyTorch models allowing for posterior approximation rather than independent models themselves.

## `fit`

All methods have a `fit` method that trains the model as well as calculates the parameters needed for model sampling and uncertainty estimation. The `fit` method always requires the `train_loader` parameter that is a `torch.utils.data.DataLoader` instance. You can additionally optionally provide a `val_loader` of the same type for validation.

Keep in mind that if you use [`LaplaceApproximation`](../laplace-approximation), which is a post hoc method primarily used to approximate posterior weight distributions for pretrained models, the `fit` method will by default just call the `compute_posterior` method without model training (more on that in the [section about Laplace approximation](../laplace-approximation)). 

## `predict`

The `predict` method is use to make predictions for test objects with uncertainty estimation by sampling a given amount of model weights and making the corresponding amount of predctions. The parameters of `predict` in all models are:

- `X: torch.Tensor`: the object for which you are making a prediction. Keep in mind that it must be a `torch.Tensor` instance of the same shape as training objects.

- `n_samples: int = 100`: the number of weight samples used to make the prediction. The default value of this parameter may vary across different methods depending on sampling speeds.

The `predict` method outputs a tuple of `torch.Tensor` object containing the mean predictions and their uncertainties.

## `sample_models`

This is the go-to method for model ensemble sampling. It takes the `n_models` parameter as input, defining the number of models to be sampled, and returns a list of sampled models as `nn.Module` instances. The sampled models are in fact just copies of the original model with perturbed weights.

## `save`

This method takes a `path` file name as input and saves the ensemble state to the specified file.

## `load`

This method takes a `path` file name as input and loads the ensemble state for the specified file.