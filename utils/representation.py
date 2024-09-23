import numpy as np

# Graphical representation
def draw_figure_lineal(iplt, x, y, x_min_max=None, y_train_pred=None, x_new=None, cost=None):
    iplt.figure(figsize=(10, 5))
    iplt.title('Graphical representation of the data set')
    iplt.plot(x, y, 'b.')
    if x_min_max is not None:
        iplt.plot(x_min_max, y_train_pred, "g-")
    if cost is not None:
        iplt.plot(x_new, cost, "rx")
    iplt.xlabel("Affected equipment")
    iplt.ylabel("Incident cost")


def draw_decision_boundary(iplt, model, reduced_features, labels):
    # Scatter plot of the data points
    iplt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', edgecolor='k', s=20)

    # Create a mesh grid for the decision boundary
    x_min, x_max = reduced_features[:, 0].min() - 1, reduced_features[:, 0].max() + 1
    y_min, y_max = reduced_features[:, 1].min() - 1, reduced_features[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Predict the class for each point in the mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    iplt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

    # Add labels and title
    iplt.set_xlabel('Principal Component 1')
    iplt.set_ylabel('Principal Component 2')
    iplt.set_title('Logistic Regression Decision Boundary')
    iplt.colorbar(label='Spam (1) or Not Spam (0)')