import logging

import matplotlib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import plot_tree

console_log = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(message)s')
console_log.setFormatter(formatter)
logger = logging.getLogger("main")
logger.addHandler(console_log)
logger.setLevel(logging.INFO)

logger.info('Starting...')

path = "/home/ichida/dev_env/ml/data/zara_challenge/zara_data_go_2019_all_dataset"

sales_stock_df = pd.read_csv(f"{path}/sales_stock.csv")
products_df = pd.read_csv(f"{path}/products.csv")
positions_df = pd.read_csv(f"{path}/positions.csv")

position_features = positions_df.groupby(['date_number', 'product_id']).agg(
    {'position': ['max', 'mean', 'min']}).reset_index()
position_features.columns = ['date_number', 'product_id', 'max_position', 'mean_position', 'position']
product_sales_stock = pd.merge(products_df, sales_stock_df, on='product_id')
groupby_columns = ['product_id', 'family_id', 'subfamily_id', 'price', 'date_number', 'color_id', 'size_id']
product_sales_stock = product_sales_stock.groupby(groupby_columns).agg({'sales': 'sum', 'stock': 'sum'}).reset_index()

all_features = pd.merge(product_sales_stock, position_features, on=['date_number', 'product_id'])
all_features.loc[:, 'product_id'] = all_features.loc[:, 'product_id'].astype('category')
all_features.loc[:, 'family_id'] = all_features.loc[:, 'family_id'].astype('category')
all_features.loc[:, 'size_id'] = all_features.loc[:, 'size_id'].astype('category')
all_features.loc[:, 'color_id'] = all_features.loc[:, 'color_id'].astype('category')
all_features = all_features.drop(columns=['product_id', 'subfamily_id'], axis=1)

test_values = all_features[all_features['date_number'] > 85]
val_values = all_features[(all_features['date_number'] > 79) & (all_features['date_number'] <= 85)]
train_features = all_features[all_features['date_number'] <= 79]


class TypeSelector(BaseEstimator, TransformerMixin):
    def __init__(self, dtype):
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include=[self.dtype])


transformer = Pipeline([
    ('features', FeatureUnion(n_jobs=1, transformer_list=[
        ('numericals', Pipeline([
            ('selector', TypeSelector(np.number)),
            ('scaler', StandardScaler())
        ])),
        # Categorical features
        ('categoricals', Pipeline([
            ('selector', TypeSelector('category')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]))
    ])),
    ('clf', xgb.XGBRegressor(objective="reg:squarederror", booster="gbtree", nthread=3, gpu_id=0,
                             tree_method='gpu_hist', n_estimators=400))
])

param_grid = {
    'clf__max_depth': np.arange(2, 10, 1)
}

logger.info(f'max deep: {np.arange(2, 10, 1)}')

randomized_mse = RandomizedSearchCV(param_distributions=param_grid, estimator=transformer, n_iter=10,
                                    scoring="neg_mean_squared_error", verbose=3, cv=3)

x, y = train_features.drop(columns=['sales', 'date_number'], axis=1), train_features['sales']
print(f'Features: {x.columns} = {x.dtypes}')
print('start fitting model...')
randomized_mse.fit(x, y)
print(randomized_mse.best_score_)
print(randomized_mse.best_estimator_)

x_test, y_test = test_values.drop(columns=['sales', 'date_number'], axis=1), test_values['sales']
preds_test = randomized_mse.best_estimator_.predict(x_test)
mse = mean_squared_error(y_test.values, preds_test)

logger.info(f'Error: {mse:0.4f}')

df_test = x_test.copy()
df_test.loc[:, 'predicted_sales'] = preds_test
df_test.loc[:, 'sales'] = y_test
df_test.loc[:, 'residual'] = df_test.loc[:, 'predicted_sales'] - df_test.loc[:, 'sales']
df_test[df_test['sales'] > 40][['sales', 'predicted_sales', 'residual']]

plot_tree(randomized_mse.best_estimator_.named_steps['clf'])
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(150, 100)
fig.savefig('tree.png')
