import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import shap

# dataset_url  = 'https://drive.google.com/file/d/1O6nQUDSHNMwey5fjR75Gxey_Bj9Z1Glj/view?usp=sharing'
data = pd.read_csv('D:/Implementation/Data/Trapelo_data.csv')
# print (data.head())
# value = data['mass_error_min'].quantile(0.98)
# data = data.replace(np.inf, value)

y = data.Critical_Zone
x = data.drop(['Critical_Zone','Room'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# Prepares a default instance of the random forest regressor
model = RandomForestRegressor()

# Fits the model on the data
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Fits the explainer
explainer = shap.Explainer(model.predict, X_test)

# Calculates the SHAP values - It takes some time
shap_values = explainer(X_test)

# Report Sharpley value by plots
shap.summary_plot(shap_values)
shap.plots.waterfall(shap_values[0])
shap.initjs()
#instace number = 4127
# shap.force_plot(explainer.expected_value, shap_values[0, :], X_sample.iloc[0, :], matplotlib=True, show=False)
# plt.savefig('force_plot.png')
shap.plots.force(shap_values[0],matplotlib=True, show=True)
shap.plots.force(shap_values[1],matplotlib=True, show=True)
shap.plots.force(shap_values[2],matplotlib=True, show=True)
shap.plots.force(shap_values[3],matplotlib=True, show=True)
shap.plots.force(shap_values[4],matplotlib=True, show=True)
shap.plots.force(shap_values[6],matplotlib=True, show=True)
shap.plots.force(shap_values[7],matplotlib=True, show=True)

print(shap_values)