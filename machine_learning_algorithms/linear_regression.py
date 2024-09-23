import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from utils.formatting.Spinner import Spinner
from utils.formatting.colors import Colors
from utils.representation import draw_figure_lineal


# Scaling the number of affected computers
def scale(df):
    df['n_affected_computers'] = df['n_affected_computers'] * 1000
    df['n_affected_computers'] = df['n_affected_computers'].astype('int')
    # Cost escalation
    df['cost'] = df['cost'] * 10000
    df['cost'] = df['cost'].astype('int')
    return df


# Cost of a security incident

# The exercise consists of predicting the cost of a security incident based on the number
# of computers that have been affected. The data set is generated randomly.


def linear_regression():
    spinner = Spinner()
    print(Colors.HEADER, Colors.UNDERLINE, "\nLINEAR REGRESSION ALGORITHM \n", Colors.ENDC)
    print(Colors.HEADER, "\nThe exercise consists of predicting the cost of a security incident based on the number\n"
                         "of computers that have been affected. In this case the data set is generated randomly in real\n"
                         "situations should use real data recollected from past experiences. \n\n", Colors.ENDC)

    input_checked = False
    num_sample = 0
    while input_checked is False:
        num_sample = int(input(
            Colors.OKCYAN + "\nEnter the number of past security incidents the data set should contain: (!max-100000000)\n"
            + Colors.ENDC))
        if 0 < num_sample:
            input_checked = True

    x = spinner.run_with_spinner(lambda: 2 * np.random.rand(num_sample, 1), 'affected computers data')
    y = spinner.run_with_spinner(lambda: 4 + 3 * x + np.random.randn(num_sample, 1), 'cost data')

    print(Colors.WARNING, "\nThe length of the data set is:", Colors.ENDC, Colors.OKGREEN, len(x), '\n', Colors.ENDC)

    data = spinner.run_with_spinner(lambda: {'n_affected_computers': x.flatten(), 'cost': y.flatten()},
                                    'data into python dictionary')
    df = spinner.run_with_spinner(lambda: pd.DataFrame(data), 'dataframe')

    print(Colors.WARNING, "\nGenerated data set", Colors.ENDC)
    print(df.head(10).to_markdown())

    # Scaling the number of affected computers
    df = spinner.run_with_spinner(lambda: scale(df), 'data scaling')

    print(Colors.WARNING, "\n1000x scaled data set", Colors.ENDC)
    print(df.head(10).to_markdown())

    # Graphical representation of the data set

    spinner.run_with_spinner(lambda: draw_figure_lineal(plt, df['n_affected_computers'], df['cost']), 'data set figure')

    # Construction of the model and adjustment of the hypothesis function
    lin_reg = LinearRegression()

    spinner.run_with_spinner(lambda: lin_reg.fit(df['n_affected_computers'].values.reshape(-1, 1), df['cost'].values),
                             'training model')

    # Theta parameter 0
    print(Colors.WARNING, "\nTheta parameter 0 (starting point of x)", Colors.ENDC)
    print(Colors.OKGREEN, round(lin_reg.intercept_, 3), Colors.ENDC)

    # Theta parameter 1
    print(Colors.WARNING, "\nTheta parameter 1 (coefficient, slope of the line)", Colors.ENDC)
    print(Colors.OKGREEN, round(float(lin_reg.coef_), 3), Colors.ENDC)

    # Prediction for the minimum and maximum value of the training data set
    x_min_max = np.array([[df["n_affected_computers"].min()], [df["n_affected_computers"].max()]])

    y_train_pred = spinner.run_with_spinner(lambda: lin_reg.predict(x_min_max), 'hypothesis function')

    # Graphical representation of the generated hypothesis function
    spinner.run_with_spinner(lambda: draw_figure_lineal(plt, df['n_affected_computers'], df['cost'],
                                                        x_min_max, y_train_pred), 'hypothesis function figure')

    # Predicting new examples
    pred_input_checked = False
    num_sample_pred = 0
    while pred_input_checked is False:
        num_sample_pred = int(input(Colors.OKCYAN + "\nEnter the number of affected equipment?\n" + Colors.ENDC))
        if 0 < num_sample_pred:
            pred_input_checked = True

    x_new = np.array([[num_sample_pred]])

    # Prediction of the cost that the incident would have

    cost = spinner.run_with_spinner(lambda: lin_reg.predict(x_new), 'cost prediction')

    print(Colors.OKBLUE, "\nThe cost of the incident for", Colors.OKGREEN, int(x_new[0][0]), Colors.OKBLUE,
          "affected equipment would be:", Colors.OKGREEN, int(cost[0]), "€\n")

    print(Colors.WARNING, "\nClose graphics tabs to continue", Colors.ENDC, )

    spinner.run_with_spinner(lambda: draw_figure_lineal(plt, df['n_affected_computers'], df['cost'], x_min_max,
                                                        y_train_pred, x_new, cost), 'hypothesis function figure')

    plt.show()
