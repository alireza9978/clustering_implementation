from sklearn.decomposition import PCA


def dimensions_reduction(data):
    pca = PCA(n_components=100, svd_solver='full')
    result = pca.fit_transform(data)
    return result
