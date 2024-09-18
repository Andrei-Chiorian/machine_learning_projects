import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression


# Cost of a security incident

# The exercise consists of predicting the cost of a security incident based on the number
# of computers that have been affected. The data set is generated randomly.
def linear_regression():
    x = 2 * np.random.rand(1000, 1)
    y = 4 + 3 * x + np.random.randn(1000, 1)

    print("The length of the data set is:", len(x))

    data = {'n_affected_computers': x.flatten(), 'cost': y.flatten()}
    df = pd.DataFrame(data)

    print(df)

    # Scaling the number of affected computers
    df['n_affected_computers'] = df['n_affected_computers'] * 1000
    df['n_affected_computers'] = df['n_affected_computers'].astype('int')
    # Cost escalation
    df['cost'] = df['cost'] * 10000
    df['cost'] = df['cost'].astype('int')
    df.head(10)

    print(df)

    # Graphical representation of the data set
    plt.figure(figsize=(10, 5))
    plt.title('Graphical representation of the data set')
    plt.plot(df['n_affected_computers'], df['cost'], "b.")
    plt.xlabel("Affected equipment")
    plt.ylabel("Incident cost")

    # Construction of the model and adjustment of the hypothesis function
    lin_reg = LinearRegression()
    lin_reg.fit(df['n_affected_computers'].values.reshape(-1, 1), df['cost'].values)

    # Theta parameter 0
    print(lin_reg.intercept_)

    # Theta parameter 1
    print(lin_reg.coef_)

    # Prediction for the minimum and maximum value of the training data set
    x_min_max = np.array([[df["n_affected_computers"].min()], [df["n_affected_computers"].max()]])
    y_train_pred = lin_reg.predict(x_min_max)

    # Graphical representation of the generated hypothesis function
    plt.figure(figsize=(10, 5))
    plt.title('Graphical representation of the generated hypothesis function')
    plt.plot(x_min_max, y_train_pred, "g-")
    plt.plot(df['n_affected_computers'], df['cost'], "b.")
    plt.xlabel("Affected equipment")
    plt.ylabel("Incident cost")

    # Predicting new examples
    x_new = np.array([[1300]])  # 1300 computers affected

    # Prediction of the cost that the incident would have
    cost = lin_reg.predict(x_new)

    print("The cost of the incident for", int(x_new[0][0]), "affected equipment would be:", int(cost[0]), "€")

    plt.figure(figsize=(10, 5))
    plt.title('Graphical representation of the prediction of the cost that the incident would have')
    plt.plot(df['n_affected_computers'], df['cost'], "b.")
    plt.plot(x_min_max, y_train_pred, "g-")
    plt.plot(x_new, cost, "rx")
    plt.xlabel("Affected equipment")
    plt.ylabel("Incident cost")

    plt.show()
