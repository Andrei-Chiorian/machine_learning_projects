from machine_learning_algorithms.linear_regression import linear_regression
from utils.formatting.colors import Colors
from machine_learning_algorithms.logistic_regression import logistic_regression

if __name__ == '__main__':

    while True:
        print(Colors.HEADER, Colors.UNDERLINE, "\nMACHINE LEARNING ALGORITHM'S \n\n", Colors.ENDC)

        print(Colors.OKBLUE, "- 1 - Linear Regression", Colors.ENDC)

        print(Colors.OKBLUE, "\n - 2 - Logistic Regression", Colors.ENDC)

        print(Colors.OKBLUE, "\n - 3 - Linear Regression", Colors.ENDC)

        print(Colors.OKBLUE, "\n - 4 - Linear Regression", Colors.ENDC)

        algorithm = input(Colors.OKCYAN + "\n¿Enter the number of the machine learning algorithm would you like to run?"
                                          "\n" + Colors.ENDC)

        if algorithm:
            match int(algorithm):
                case 1:
                    linear_regression()
                    print(Colors.OKCYAN, "\n\nLinear Regression Algorithm example completed", Colors.ENDC)
                    print(Colors.FAIL, "\n\n-------------------------------------------------------------------------"
                                       "--------------", Colors.ENDC)
                case 2:
                    logistic_regression()
                    print(Colors.OKCYAN, "\n\nLogistic Regression Algorithm example completed", Colors.ENDC)
                    print(Colors.FAIL, "\n\n-------------------------------------------------------------------------"
                                       "--------------", Colors.ENDC)
                case 3:
                    print(Colors.OKCYAN, "Opción 3 seleccionada", Colors.ENDC)
                case _:
                    print(Colors.OKCYAN, "Opción no válida", Colors.ENDC)
