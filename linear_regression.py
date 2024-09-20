import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from colors import Colors

# Colors for the output


# Cost of a security incident

# The exercise consists of predicting the cost of a security incident based on the number
# of computers that have been affected. The data set is generated randomly.
def linear_regression():
    print(Colors.HEADER, Colors.UNDERLINE, "\nLINEAR REGRESSION ALGORITHM'S \n", Colors.ENDC)
    print(Colors.HEADER, "\nThe exercise consists of predicting the cost of a security incident based on the number\n"
                         "of computers that have been affected. In this case the data set is generated randomly in real\n"
                         "situations should use real data recollected from past experiences. \n\n", Colors.ENDC)

    num_sample = int(input(Colors.OKCYAN + "\nEnter the number of past security incidents the data set should contain? (!max-100000000)\n" + Colors.ENDC))

    x = 2 * np.random.rand(num_sample, 1)
    y = 4 + 3 * x + np.random.randn(num_sample, 1)

    print(Colors.WARNING, "\nThe length of the data set is:", Colors.ENDC, Colors.OKGREEN, len(x), '\n', Colors.ENDC)

    data = {'n_affected_computers': x.flatten(), 'cost': y.flatten()}
    df = pd.DataFrame(data)

    print(Colors.WARNING, "\nGenerated data set" , Colors.ENDC)
    print(df)

    # Scaling the number of affected computers
    df['n_affected_computers'] = df['n_affected_computers'] * 1000
    df['n_affected_computers'] = df['n_affected_computers'].astype('int')
    # Cost escalation
    df['cost'] = df['cost'] * 10000
    df['cost'] = df['cost'].astype('int')
    df.head(10)

    print(Colors.WARNING, "\n1000x scaled data set", Colors.ENDC)
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
    print(Colors.WARNING, "\nTheta parameter 0 (starting point of x)", Colors.ENDC)
    print(Colors.OKGREEN, round(lin_reg.intercept_, 3), Colors.ENDC)

    # Theta parameter 1
    print(Colors.WARNING, "\nTheta parameter 1 (coefficient, slope of the line)", Colors.ENDC)
    print(Colors.OKGREEN, round(float(lin_reg.coef_), 3), Colors.ENDC)

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
    x_new = np.array([[int(input(Colors.OKCYAN + "\nEnter the number of affected equipment?\n" + Colors.ENDC))]])

    # Prediction of the cost that the incident would have
    cost = lin_reg.predict(x_new)

    print(Colors.OKBLUE, "\nThe cost of the incident for", Colors.OKGREEN, int(x_new[0][0]), Colors.OKBLUE,
          "affected equipment would be:", Colors.OKGREEN, int(cost[0]), "€\n")

    print(Colors.WARNING, "\nClose graphics tabs to continue", Colors.ENDC,)

    plt.figure(figsize=(10, 5))
    plt.title('Graphical representation of the prediction of the cost that the incident would have')
    plt.plot(df['n_affected_computers'], df['cost'], "b.")
    plt.plot(x_min_max, y_train_pred, "g-")
    plt.plot(x_new, cost, "rx")
    plt.xlabel("Affected equipment")
    plt.ylabel("Incident cost")
    plt.show()
