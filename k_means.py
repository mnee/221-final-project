import numpy as np

"""
    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
"""
def k_means_vectorized(features, k = 6, num_iters = 100):
    N, D = features.shape
    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N)
    print(N*k)
    for n in range(num_iters):
        prev = centers.copy()
        means = np.asarray([np.array([]) for i in range(k)])
        feat = np.repeat(features, (k), axis = 0)
        cent_init = np.tile(centers, (len(features), 1))
        # print('cent_init:{}, \n feat:{}'.format(cent_init.shape, feat.shape))
        # print('cent_type:{}, new type:{}'.format(type(cent_init[0, 0]), type(cent_init[0, 13])))
        diff = np.linalg.norm(cent_init - feat, axis = 1).reshape(N, k)
        assignments = np.argmin(diff, axis = 1)
        for i in range(k):
            want = features[assignments == i,:]
            if len(want) == 0: centers[i] = 0
            else: centers[i] = sum(want)/float(len(want))
        if np.array_equal(centers, prev): break

    return assignments, centers
