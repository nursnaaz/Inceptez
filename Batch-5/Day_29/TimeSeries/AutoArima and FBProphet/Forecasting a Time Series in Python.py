import numpy as np
import pyramid as pm
from pyramid.datasets import load_wineind

# this is a dataset from R
wineind = load_wineind().astype(np.float64)

# fit stepwise auto-ARIMA
stepwise_fit = pm.auto_arima(wineind, start_p=1, start_q=1,
                             max_p=3, max_q=3, m=12,
                             start_P=0, seasonal=True,
                             d=1, D=1, trace=True,
                             error_action='ignore',  # don't want to know if an order does not work
                             suppress_warnings=True,  # don't want convergence warnings
                             stepwise=True)  # set to stepwise

stepwise_fit.summary()

from pyramid.arima import auto_arima
from pyramid.datasets import load_lynx
import numpy as np

# For serialization:
from sklearn.externals import joblib
import pickle

# Load data and fit a model
y = load_lynx()
arima = auto_arima(y, seasonal=True)

# Serialize with Pickle
with open('arima.pkl', 'wb') as pkl:
    pickle.dump(arima, pkl)

# You can still make predictions from the model at this point
arima.predict(n_periods=5)

# Now read it back and make a prediction
with open('arima.pkl', 'rb') as pkl:
    pickle_preds = pickle.load(pkl).predict(n_periods=5)

# Or maybe joblib tickles your fancy
joblib.dump(arima, 'arima.pkl')
joblib_preds = joblib.load('arima.pkl').predict(n_periods=5)

# show they're the same
np.allclose(pickle_preds, joblib_preds)


import pyramid as pm
from pyramid.datasets import load_wineind

y = load_wineind()
train, test = y[:125], y[125:]

# Fit an ARIMA
arima = pm.ARIMA(order=(1, 1, 2), seasonal_order=(0, 1, 1, 12))
arima.fit(y)



arima.add_new_observations(test)  # pretend these are the new ones


