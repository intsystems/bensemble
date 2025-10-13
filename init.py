from .core.base import BaseBayesianEnsemble
from .methods.variational_inference import VariationalInference
from .methods.laplace_approximation import LaplaceApproximation
from .methods.variational_renyi import VariationalRenyi
from .core.utils import compute_uncertainty, enable_dropout

class BayesianEnsembleLibrary:
    """
    Унифицированный API для байесовского ансамблирования
    """
    
    def __init__(self):
        self.methods = {
            'VI': VariationalInference,
            'Laplace': LaplaceApproximation,
            'VR': VariationalRenyi,
        }
        self.current_ensemble = None
    
    def load_model(self, model):
        """Загрузка модели"""
        self.model = model
    
    def load_data(self, X_train, y_train, X_val=None, y_val=None, batch_size=32):
        """Загрузка данных"""
        self.train_loader = self._create_data_loader(X_train, y_train, batch_size)
        if X_val is not None and y_val is not None:
            self.val_loader = self._create_data_loader(X_val, y_val, batch_size)
        else:
            self.val_loader = None
    
    def _create_data_loader(self, X, y, batch_size):
        dataset = torch.utils.data.TensorDataset(X, y)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def fit_ensemble(self, method: str = 'VI', **kwargs):
        """Обучение ансамбля выбранным методом"""
        if method not in self.methods:
            raise ValueError(f"Method {method} not supported. Available: {list(self.methods.keys())}")
        
        ensemble_class = self.methods[method]
        self.current_ensemble = ensemble_class(self.model, **kwargs)
        
        history = self.current_ensemble.fit(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            **kwargs
        )
        
        return history
    
    def predict(self, X_test, n_samples=100):
        """Предсказание с оценкой неопределенности"""
        if self.current_ensemble is None:
            raise RuntimeError("No ensemble fitted. Call fit_ensemble() first.")
        
        return self.current_ensemble.predict(X_test, n_samples)
    
    def sample_models(self, n_models=10):
        """Сэмплирование моделей из ансамбля"""
        if self.current_ensemble is None:
            raise RuntimeError("No ensemble fitted. Call fit_ensemble() first.")
        
        return self.current_ensemble.sample_models(n_models)
    
    def save_ensemble(self, path: str):
        """Сохранение ансамбля"""
        if self.current_ensemble is None:
            raise RuntimeError("No ensemble to save.")
        self.current_ensemble.save(path)
    
    def load_ensemble(self, path: str, method: str):
        """Загрузка ансамбля"""
        if method not in self.methods:
            raise ValueError(f"Method {method} not supported.")
        
        ensemble_class = self.methods[method]
        self.current_ensemble = ensemble_class(self.model)
        self.current_ensemble.load(path)