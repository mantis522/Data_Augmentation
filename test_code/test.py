from sklearn.utils.extmath import softmax
import numpy as np

X = np.array([[50, 60, 66]])
print(softmax(X.astype(np.double)))