from sklearn.utils.extmath import softmax
import numpy as np

X = np.array([[12.5, 11.875, 10.625], [8.125, 12.5, 8.125], [12.5, 7.5, 13.75]])
print(softmax(X.astype(np.double)))