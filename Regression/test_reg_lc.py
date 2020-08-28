from __future__ import print_function
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV


def fit_and_learn(data, model):
    total_samples = len(df.index)
    test_portion = 0.2
    step = total_samples // 50  # integer division

    num_samples = []
    train_rmses = []
    test_rmses = []

    # Gradually use more data to get the learning curve
    for num_samp in (total_samples if n>total_samples else n for n in range(step, total_samples+step, step)):
        # Divide the data in training and test set
        num_samples.append(num_samp)
        sample_idxs = random.sample(range(0, total_samples), num_samp)
        x_train, x_test, y_train, y_test = train_test_split(data['x'].values[sample_idxs],
                                                            data['y'].values[sample_idxs],
                                                            test_size=test_portion)

        x_train = x_train.reshape(-1,1)  # 1D (N) row -> 2D (N,1) - a column
        x_test = x_test.reshape(-1,1)  # or x_test[:, np.newaxis]

        model.fit(x_train,y_train)  # modifies the model!

        y_test_pred = model.predict(x_test)
        test_rmse = sqrt(mean_squared_error(y_test, y_test_pred))
        test_rmses.append(test_rmse)

        y_train_pred = model.predict(x_train)
        train_rmse = sqrt(mean_squared_error(y_train, y_train_pred))
        train_rmses.append(train_rmse)

    # Plot the learning curves
    axes = plt.gca()
    axes.set_title('Learning curves')
    plt.plot(num_samples, train_rmses, color=u'#1f77b4')
    axes.text(0.7, 0.95, 'train', transform=axes.transAxes, color=u'#1f77b4', fontsize=10)
    plt.plot(num_samples, test_rmses, color=u'#ff7f0e')
    axes.text(0.7, 0.90, 'test', transform=axes.transAxes, color=u'#ff7f0e', fontsize=10)
    axes.set_xlabel('number of samples')
    axes.set_ylabel('RMSE')
    plt.show()

    # Return the last RMSEs
    return train_rmses[-1], test_rmses[-1]


# Read and sort the data
df = pd.read_csv('../data/test.csv', delimiter=';', decimal=',', index_col=0)
df.sort_index(inplace=True)


# Linear regression model
lin_reg_model = LinearRegression()

# Fit the model and plot the learning curves
train_rmse, test_rmse = fit_and_learn(df, lin_reg_model)

# Plot the original data and the linear regression line
df.plot(x='x', y='y', style='o', ms=3, title='Original data and linear regression')  # , legend=False
axes = plt.gca()
x_vals = np.array(axes.get_xlim())
y_vals = lin_reg_model.coef_ * x_vals + lin_reg_model.intercept_  # coef_ is line slope.
plt.plot(x_vals, y_vals, '--')
axes.text(0.5, 0.95, 'RMSE train = ' + str(train_rmse), transform=axes.transAxes, fontsize=10)
axes.text(0.5, 0.90, 'RMSE test = ' + str(test_rmse), transform=axes.transAxes, fontsize=10)
plt.show()


def polynomial_regression( data, deg ):
    poly_model = Pipeline([
        ("poly", PolynomialFeatures(degree=deg)),
        ("std_scaler", StandardScaler()),  # standardize features by subtracting mean and scaling to unit variance
        ("lin_reg", LinearRegression())  # linear model with polynomial features
    ])

    # Fit the model and plot the learning curves
    train_rmse, test_rmse = fit_and_learn(data, poly_model)

    # Plot the original data and the polynomial regression line
    data.plot(x='x', y='y', style='o', ms=3, title='Original data and polynomial regression degree '+str(deg))
    x_vals = data['x'].values
    y_vals = poly_model.predict(x_vals[:, np.newaxis])
    plt.plot(x_vals, y_vals, '--')
    axes = plt.gca()
    axes.text(0.5, 0.95, 'RMSE train = ' + str(train_rmse), transform=axes.transAxes, fontsize=10)
    axes.text(0.5, 0.90, 'RMSE test = ' + str(test_rmse), transform=axes.transAxes, fontsize=10)
    plt.show()

    return poly_model


# Try different polynomial features degrees. Still linear model though.
polynomial_regression(df, 2)

poly_model20 = polynomial_regression(df, 20)
X_plot = np.linspace(-0.5, 20.5, 800)
y_plot = poly_model20.predict(X_plot.reshape(-1, 1))
df.plot(x='x', y='y', style='o', ms=3, title='Original data and polynomial regression degree 20 test')
plt.plot(X_plot, y_plot, '--')
plt.show()

# Can we use GridSearchCV to determine best polynomial degree?
# Makes more sense with several parameters, but anyway...
# For GridSearchCV it is mandatory to use a pipeline, because it guarantees
# cross-validation and scaling synchronization: "the splitting of the data-set
# during cross-validation should be done before doing any pre-processing",
# otherwise cross-validation set will contain information about the whole
# data set (which we don't want), because whole data set is used for scaling.
poly_pipe = make_pipeline(PolynomialFeatures(),
                          StandardScaler(),
                          LinearRegression())

param_grid = { 'polynomialfeatures__degree': [i for i in range(2, 20)] }

grid_search = GridSearchCV(poly_pipe, param_grid=param_grid,
                           scoring='neg_mean_squared_error', verbose=1)

grid_search.fit(df['x'].values[:,np.newaxis], df['y'].values)
print("grid_search.best_params_ = ", grid_search.best_params_)
best_poly_model = grid_search.best_estimator_

# After finding the optimal hyper-parameters using cross-validation -
# re-train with the whole training set, if GridSearchCV is initialized
# with refit=True (which is the default)


# Regularization (L2 Ridge(alpha) or L1 Lasso(alpha) instead of LinearRegression())?
