from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import mahalanobis
from scipy import stats

# Function to segment the image using K-means clustering
def segmention(cube, k):
    # Reshape the image into a 2D array (height * width, bands)
    reshaped_cube = cube.reshape(-1, cube.shape[2])

    # Standardize the data (mean = 0 and variance = 1)
    scaler = StandardScaler()
    scaled_image = scaler.fit_transform(reshaped_cube)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    clustered = kmeans.fit_predict(scaled_image)

    # Reshape the clustered data back to the original image shape
    clustered_cube = clustered.reshape(cube.shape[0], cube.shape[1])

    # Display the original image and the clustered image
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cube[:, :, 0])
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Segmented Image (K-means)")
    plt.imshow(clustered_cube, cmap='viridis')  # You can choose any colormap you like
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return clustered_cube

# Function to calculate Mahalanobis distances
def calculate_mahalanobis_distances(pixels, mean_vec, cov):
    inv_cov = np.linalg.inv(cov)
    distances = np.array([mahalanobis(x, mean_vec, inv_cov)**2 for x in pixels])
    return distances

# Function to create a mixture F-distribution CDF
def create_mixture_f_cdf(params, L):
    def mixture_f_cdf(d):
        w, nu1, nu2 = params
        # Compute the scaled arguments
        arg1 = (d / L) * (nu1 / (nu1 - 2))
        arg2 = (d / L) * (nu2 / (nu2 - 2))

        # Compute the CDF of the first F-distribution component
        component1 = stats.f.cdf(arg1, dfn=L, dfd=nu1)
        # Compute the CDF of the second F-distribution component
        component2 = stats.f.cdf(arg2, dfn=L, dfd=nu2)

        # Compute the weighted sum of the two components
        mixture_cdf = w * component1 + (1 - w) * component2

        return mixture_cdf
    return mixture_f_cdf

# Function to calculate exceedance metric
def calc_exeedance_metric(data_d, data_exceedance, mixture_d, mixture_exceedance, k=100):
    epsilon = (1 / data_d.shape[0])  # Avoiding extreme outliers
    prob_points = np.logspace(np.log10(10 * epsilon), np.log10(1), num=k)
    exceedance_metric_arr = np.zeros(k)

    for i, prob in enumerate(prob_points):
        # Find the closest value index for data and mixture
        abs_diffs_data = np.abs(data_exceedance - prob)
        index_data = abs_diffs_data.argmin()
        data_prob = data_exceedance[index_data]
        abs_diffs_mixture = np.abs(mixture_exceedance - data_prob)
        index_mixture = abs_diffs_mixture.argmin()

        # Check if the closest index is actually within a reasonable range
        if abs_diffs_data[index_data] > epsilon:
            inv_data_exceedance = np.nan  # Mark as not found
        else:
            inv_data_exceedance = data_d[index_data]

        if abs_diffs_mixture[index_mixture] > epsilon:
            inv_mixture_exceedance = np.nan  # Mark as not found
        else:
            inv_mixture_exceedance = mixture_d[index_mixture]

        if not np.isnan(inv_data_exceedance) and not np.isnan(inv_mixture_exceedance):
            exceedance_metric_arr[i] = abs(inv_data_exceedance - inv_mixture_exceedance)
        else:
            exceedance_metric_arr[i] = np.nan  # In case of invalid data

    # Exclude NaNs from the sum
    valid_metrics = exceedance_metric_arr[~np.isnan(exceedance_metric_arr)]
    return np.sum(valid_metrics)

# Function to calculate exceedance metric by params
def exceedance_metric_by_params(params, L, mixture_d, data_d, data_exeedance):
    mixture_f = create_mixture_f_cdf(params, L)
    mixture_exeedance = 1 - mixture_f(mixture_d)
    exeedance_metric = calc_exeedance_metric(data_d, data_exeedance, mixture_d, mixture_exeedance)
    # Debug prints
    print(f"params = {params}")
    print(f"exeedance_metric = {exeedance_metric}")
    return exeedance_metric

# Function to find the best mixture
def find_best_mixture(data_d, data_exeedance, L, method):
    mixture_d = np.linspace(0, np.max(data_d), 1000000)

    if method == "grid_search" or method == "grid_and_optimization":
        best_params = 0, 0, 0

        w_arr = np.linspace(0.5, 1, 5)
        nu1_arr = nu2_arr = [
            1000, 500, 100, 90, 80, 70,
            60, 50, 45, 40, 35, 30, 29,
            28, 27, 26, 25, 24, 23, 22,
            21, 20, 19, 18, 17, 16, 15,
            14, 13, 12, 11, 10, 9, 8, 7, 6, 5
        ]
        min_exeedance_matric = float('inf')

        for w in w_arr:
            for nu1 in nu1_arr:
                for nu2 in nu2_arr:
                    params = w, nu1, nu2
                    exeedance_metric = exceedance_metric_by_params(params, L, mixture_d, data_d, data_exeedance)
                    if exeedance_metric < min_exeedance_matric:
                        min_exeedance_matric = exeedance_metric
                        best_params = params

    if method == "optimization" or method == "grid_and_optimization":
        initial_guess = best_params if method == "grid_and_optimization" else [0.5, 100, 5]
        bounds = [(0, 1), (2, 1000), (2, 1000)]  # bounds for w, nu1, nu2
        tol = 1e-10

        result = minimize(exceedance_metric_by_params, initial_guess, args=(L, mixture_d, data_d, data_exeedance), method='Nelder-Mead', bounds=bounds, tol=tol)
        if result.success:
            best_params = result.x
        else:
            raise ValueError("Optimization failed")

    return best_params

# Custom distribution class for Mixture Multivariate T
class MixtureMultivarientT(stats.rv_continuous):
    def __init__(self, mean, cov, w, nu1, nu2):
        scale1 = cov * ((nu1 - 2) / nu1)
        scale2 = cov * ((nu2 - 2) / nu2)
        self.w = w
        self.t1 = stats.multivariate_t(loc=mean, shape=scale1, df=nu1)
        self.t2 = stats.multivariate_t(loc=mean, shape=scale2, df=nu2)

        super().__init__(name='MixtureMultivarientT')

    def _pdf(self, x):
        # Define your custom PDF here
        return self.w * self.t1.pdf(x) + (1 - self.w) * self.t2.pdf(x)

    def rvs(self, size=1, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)

        # Generate samples
        samples = []
        for _ in range(size):
            if np.random.rand() < self.w:
                samples.append(self.t1.rvs(random_state=random_state))
            else:
                samples.append(self.t2.rvs(random_state=random_state))
        return np.array(samples)[0] if size == 1 else np.array(samples)
