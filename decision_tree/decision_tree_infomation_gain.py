import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

X_train = [
    [1, 1, 1],
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 1],
    [1, 1, 1],
    [1, 1, 0],
    [0, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 1, 0]
]

y_train = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 0])

def entropy(p):
    if p == 0 or p == 1:
        return 0
    else:
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    
def split_indices(X, index_feature):
    left_indices = []
    right_indices = []
    for i, x in enumerate(X):
        if x[index_feature] == 1:
            left_indices.append(i)
        if x[index_feature]:
           right_indices.append(i)
    return left_indices, right_indices

def weighted_entropy(X,y,left_indices,right_indices):
    """
    This function takes the splitted dataset, the indices we chose to split and␣
    ↪→returns the weighted entropy.
    """
    w_left = len(left_indices)/len(X)
    w_right = len(right_indices)/len(X)
    p_left = sum(y[left_indices])/len(left_indices)
    p_right = sum(y[right_indices])/len(right_indices)
    weighted_entropy = w_left * entropy(p_left) + w_right * entropy(p_right)
    return weighted_entropy


left_indices, right_indices = split_indices(X_train, 0)

def information_gain(X_train, y_train,left_indices, right_indices):
    """
    Here, X has the elements in the node and y is theirs respectives classes
    """
    p_node = sum(y_train) / len(y_train)
    h_node = entropy(p_node)
    wt_entropy = weighted_entropy(X_train, y_train, left_indices, right_indices)
    print(wt_entropy)
    return h_node - wt_entropy


info_gain = information_gain(X_train, y_train, left_indices, right_indices)


# try:
#     info_gain = information_gain(X_train, y_train, left_indices, right_indices)
# except ValueError: 
#     print(ValueError)

print(info_gain)


for i, feature_name in enumerate(['Ear Shape', 'Face Shape', 'Whiskers']):
    left_indices, right_indices = split_indices(X_train, i)
    i_gain = information_gain(X_train, y_train, left_indices, right_indices)
    # print(f'feature: {feature_name}, information gain if we split the root node␣
    # ↪→using this feature: {i_gain:.2f}')