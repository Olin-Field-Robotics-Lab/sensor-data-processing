from h5_processing import *

pcs, weights, train_means = train_PCA(5555, 2)
print(weights)
visualize_predictions(5550, pcs, weights, train_means)