import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from lazypredict.Supervised import LazyRegressor

data = pd.read_csv('dataset\StudentScore.xls')

# print(data.info())

# sns.histplot(data['math score'])
# plt.title('Distribution of math score')
# plt.savefig('Math score')

# Set features and target
target = 'math score'
x = data.drop(target, axis=1)
y = data[target]

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=42)

# print(data['gender'].unique())

# data[data['parental level of education'] == 'some high school'] = 'high school'
# print(data['parental level of education'].unique())
# print(data['lunch'].unique())

# Categorical Data: Features "reading score" & "writing score"
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
])
# sample_data = num_transformer.fit_transform(x_train[['reading score', 'writing score']])
# for i, j in zip(x_train[['reading score', 'writing score']].values, sample_data):
#     print('Before {} After {}'.format(i, j))

# Ordinal + Binary Data: Features "parental level of education", "gender", "lunch", "test preparation course"
education_values = ["some high school", 
                    "high school", 
                    "some college", 
                    "associate's degree", 
                    "bachelor's degree",
                    "master's degree"
]
gender_values = data['gender'].unique()
lunch_values = data['lunch'].unique()
prep_values = data['test preparation course'].unique()
ord_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler', OrdinalEncoder(categories=[education_values, gender_values, lunch_values, prep_values])),
])
# sample_data = ord_transformer.fit_transform(x_train[["parental level of education"]])
# for i, j in zip(x_train[["parental level of education"]].values, sample_data):
#     print('Before {} After {}'.format(i, j))

# Nominal Data: Features "race/ethnicity"
nom_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])
# sample_data = nom_transformer.fit_transform(x_train[["race/ethnicity"]])
# for i, j in zip(x_train[["race/ethnicity"]].values, sample_data):
#     print('Before {} After {}'.format(i, j))

preprocessor = ColumnTransformer(transformers=[
    ("num_features", num_transformer, ["reading score", "writing score"]),
    ("ord_features", ord_transformer, ["parental level of education", "gender", "lunch", "test preparation course"]),
    ("nom_features", nom_transformer, ["race/ethnicity"])
])

param_grid = {
    "regressor__n_estimators": [50, 100, 200],
    "regressor__criterion": ["squared_error", "absolute_error", "friendman_mse"],
    "regressor__max_depth": [None, 5, 10],
    "preprocessor__num_features__imputer__strategy": ["mean", "median"],
    "preprocessor__nom_features__scaler__min_frequency": [None, 0.9]

}

reg = Pipeline(steps=[
    ('preprocessor', preprocessor),
    # ('regressor', RandomForestRegressor())
])

# Use LazyRegressor
x_train = reg.fit_transform(x_train)
x_test = reg.transform(x_test)
reg_2 = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
models, predictions = reg_2.fit(x_train, x_test, y_train, y_test)
print(models)

# # grid_search = GridSearchCV(estimator=reg, param_grid=param_grid, cv=5, verbose=1)
# # Use RandomizedSearchCV
# grid_search = RandomizedSearchCV(estimator=reg, param_distributions=param_grid, cv=5, verbose=1, n_iter=20)
# grid_search.fit(x_train, y_train)
# y_pred = grid_search.predict(x_test)

# # reg.fit(x_train, y_train)
# # y_pred = reg.predict(x_test)
# print("R2: {}".format(r2_score(y_test, y_pred)))
# print("MAE: {}".format(mean_absolute_error(y_test, y_pred)))
# print("MSE: {}".format(mean_squared_error(y_test, y_pred)))

# print(grid_search.best_params_)
# # # for i, j in zip(y_test, y_pred):

# # print(reg["regressor"].coef_)
# # print(reg['regressor'].intercept_)



