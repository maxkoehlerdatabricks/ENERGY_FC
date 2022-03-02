import pickle
import copy

import prophet

import mlflow
import mlflow.pyfunc
import mlflow.sklearn

from mlflow.models.signature import infer_signature

class EnergyConsumption(mlflow.pyfunc.PythonModel):
  def __init__(self, col_mapping={'y': 'y', 'ds': 'ds'}, params={}, xreg_list=[]):
    self._col_mapping = col_mapping
    self._inv_col_mapping = {v: k for k, v in col_mapping.items()}
    self._params = params
    self._xreg_list = xreg_list
    
  def _predict(self, df):
    df = copy.deepcopy(df)
    df.columns = [self._col_mapping[c] if c in self._col_mapping else c for c in df.columns]
    
    out = self._model.predict(df)
    out = out[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    out.columns = [self._inv_col_mapping[c] if c in self._inv_col_mapping else c for c in out.columns]
    return out
    
  def predict(self, context, df):
    out = self._predict(df)
    return out
  
  def fit(self, df):
    orig_columns = df.columns
    df.columns = [self._col_mapping[c] if c in self._col_mapping else c for c in df.columns]
    self._model = prophet.Prophet(**self._params)
    for xreg in self._xreg_list:
      self._model = self._model.add_regressor(xreg)
      
    self._model = self._model.fit(df)
    
    df.columns = orig_columns
    self._signature = infer_signature(df.drop(['energy_consumption'], axis=1), self._predict(df))
    return self
  
  def log_model(self, training_set=None):
    conda_env = self._conda_add_dependencies(
      conda_dep=['pip', 'cloudpickle=1.3.0'],
      pip_dep=['prophet=={}'.format(prophet.__version__)]
    )
    
    if training_set:
      from databricks.feature_store import FeatureStoreClient
      fs = FeatureStoreClient()
      fs.log_model(
        artifact_path='model',
        flavor=mlflow.pyfunc,
        conda_env=conda_env,
        signature=self._signature,
        model=self,
        training_set=training_set,
      )
    
    mlflow.pyfunc.log_model(
      artifact_path='model',
      python_model=self,
      conda_env=conda_env,
      signature=self._signature
    )
    return self
    
  @staticmethod
  def _conda_add_dependencies(conda_dep=[], pip_dep=[]):
    """
    Utility function for adding custome dependendencies.
    """
    dep = mlflow.sklearn.get_default_conda_env()['dependencies']
    
    dep += conda_dep
    for d in dep:
      if isinstance(d, dict):
        d['pip'] += pip_dep
          
    out = mlflow.sklearn.get_default_conda_env()
    out['dependencies'] = dep
    return out