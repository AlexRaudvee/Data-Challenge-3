# Data science and numerical computation libraries
import numpy as np                                              # NumPy for array and matrix operations

# Skimage for image processing and segmentation algorithms
import skimage.graph as graph                                   # Graph-based algorithms from skimage
from scipy import ndimage as ndi                                # NDI for gradient understanding on the image
from skimage.color import rgb2gray                              # Color and grayscale conversions
from skimage.segmentation import slic, random_walker            # Superpixel and segmentation algorithms
from skimage.segmentation import felzenszwalb, quickshift       # Superpixel and segmentation algorithms
from skimage.segmentation import watershed                      # Superpixel and segmentation algorithms
from skimage.util import img_as_ubyte                           # Conversion utility for images
from skimage.filters.rank import entropy                        # Entropy filter for local image statistics
from skimage.feature import peak_local_max                      # Peak detection in images
from skimage.morphology import disk                             # Disk-shaped structuring elements for morphology operations

# Machine learning algorithms from scikit-learn
from sklearn.cluster import SpectralClustering, MeanShift       # Spectral clustering for grouping superpixels
from sklearn.neighbors import KNeighborsClassifier              # K-Nearest Neighbors and Mean Shift clustering
from sklearn.metrics.pairwise import cosine_similarity          # Cosine similarity for distance metrics


class PLASPIX:
    """
    The PLASPIX class is designed to handle the generation of superpixels from an image and
    propagate labels across those superpixels using various label propagation algorithms.

    Attributes
    -----------
    superpix_model : str
        The name of the superpixel model to be used. Must be one of the following:
        'SLIC', 'QUICKSHIFT', 'FELZENSZWALB', 'WATERSHED', or 'ENTROPY'.

    label_propagation_model : str
        The name of the label propagation algorithm to be used. Must be one of the following:
        'DIFFUSION', 'GRAPH', 'DISTANCE', 'RANDOM_WALK', 'KNN', 'MEAN_SHIFT', or 'SPECTRAL'.

    Methods
    --------
    generate_superpixels(img):
        Generates superpixels from an input image based on the chosen superpixel algorithm.

    propagate_labels(superpixels, dot_coords, dot_labels, img):
        Propagates labels across superpixels using the specified label propagation algorithm.
    """

    def __init__(self, superpix_model: str, label_propagation_model: str):
        self.superpix_model = superpix_model
        self.label_propagation_model = label_propagation_model

    # Method to generate superpixels based on the model choice
    def generate_superpixels(self, img):
        """
        Generates superpixels from an input image based on the selected superpixel algorithm.

        This method generates superpixels by applying one of the five different algorithms:
        SLIC, QUICKSHIFT, FELZENSZWALB, WATERSHED, or ENTROPY. The algorithm used is determined
        by the `superpix_model` attribute of the class instance.

        Parameters
        -----------
        img : ndarray
            The input image as a NumPy array, typically in RGB or grayscale format, from which
            superpixels will be generated.

        Returns
        --------
        superpixels : ndarray
            An array representing the generated superpixels. The structure and format of the
            output depend on the specific algorithm used, but it typically contains a label map
            where each pixel is assigned a superpixel label.

        Raises
        -------
        ValueError:
            If `superpix_model` is not one of 'SLIC', 'QUICKSHIFT', 'FELZENSZWALB', 'WATERSHED',
            or 'ENTROPY', the method raises a `ValueError` indicating that the specified model is unknown.

        Superpixel Models
        ------------------
        - 'SLIC': Simple Linear Iterative Clustering, which generates compact and uniform superpixels.
        - 'QUICKSHIFT': Uses a mode-seeking algorithm to generate superpixels based on pixel intensity
        gradients and color similarities, often producing larger superpixels with irregular shapes.
        - 'FELZENSZWALB': A graph-based segmentation method that creates irregular superpixels
        based on color similarity and proximity.
        - 'WATERSHED': Treats the image as a topographic surface and forms superpixels based on the
        gradient of intensity, dividing regions using watershed lines.
        - 'ENTROPY': A method that utilizes entropy-based measures to group pixels, forming superpixels
        by optimizing the information content within each region.

        Example
        --------
        ```
        # Assuming `self.superpix_model` is set to 'SLIC'
        superpixels = self.generate_superpixels(img)

        # For different models, change `self.superpix_model` accordingly, e.g., 'QUICKSHIFT'
        superpixels = self.generate_superpixels(img)
        ```
        """

        if self.superpix_model == 'SLIC':
            return self._slic_superpixels(img)
        elif self.superpix_model == 'QUICKSHIFT':
            return self._quickshift_superpixels(img)
        elif self.superpix_model == 'FELZENSZWALB':
            return self._felzenszwalb_superpixels(img)
        elif self.superpix_model == 'WATERSHED':
            return self._watershed_superpixels(img)
        elif self.superpix_model == 'ENTROPY':
            return self._entropy_superpixels(img)
        else:
            raise ValueError(f"Unknown superpixel model: {self.superpix_model}")


    # Method to propagate labels based on the model choice
    def propagate_labels(self, superpixels, dot_coords, dot_labels, img):
        """
        Propagates labels across superpixels using the specified label propagation algorithm.

        This method assigns or refines labels across the superpixels of an image based on the
        selected label propagation model. The choice of model is controlled by the
        `label_propagation_model` attribute of the class instance.

        Parameters
        -----------
        superpixels : ndarray
            A 2D or 3D array representing the superpixels of the image. This could be a label
            map where each pixel is assigned a superpixel ID.

        superpixel_labels : ndarray
            A 1D or 2D array containing the initial labels assigned to each superpixel. This can
            be used as the starting point for the label propagation process.

        img : ndarray
            The original input image (as a NumPy array) in which label propagation will be applied.

        Returns
        --------
        propagated_labels : ndarray
            An array containing the propagated labels after applying the specified propagation
            algorithm. The structure and shape of this array will depend on the input and
            the chosen propagation model.

        Raises
        -------
        ValueError:
            If `label_propagation_model` is not one of 'DIFFUSION', 'GRAPH', 'DISTANCE', 'RANDOM_WALK',
            'KNN', 'MEAN_SHIFT', or 'SPECTRAL', the method raises a `ValueError` indicating that the
            specified model is unknown.

        Label Propagation Models
        -------------------------
        - 'DIFFUSION': Uses diffusion-based label propagation, which spreads labels across
        neighboring superpixels in a manner that mimics heat diffusion.
        - 'GRAPH': Uses graph-based label propagation, which relies on building a graph from
        the superpixels and propagating labels based on graph connectivity and similarity.
        - 'DISTANCE': Propagates labels based on the distance between superpixels, where labels
        are propagated to closer superpixels first.
        - 'RANDOM_WALK': Applies a random walk algorithm where labels are propagated across
        superpixels by treating the label propagation problem as a probabilistic walk on a graph.
        - 'KNN': Propagates labels using the k-nearest neighbors (KNN) algorithm, where labels
        are influenced by nearby labeled superpixels.
        - 'MEAN_SHIFT': Utilizes the mean shift algorithm for label propagation, where labels
        shift toward regions of high density in feature space.
        - 'SPECTRAL': Uses spectral clustering-based label propagation, which leverages
        eigenvalue-based methods to spread labels through the superpixels based on spectral similarities.

        Example
        --------
        ```
        # Assuming `self.label_propagation_model` is set to 'DIFFUSION'
        propagated_labels = self.propagate_labels(superpixels, superpixel_labels, img)

        # For different models, change `self.label_propagation_model` accordingly, e.g., 'RANDOM_WALK'
        propagated_labels = self.propagate_labels(superpixels, superpixel_labels, img)
        ```
        """

        # Step 1: Assign initial labels to superpixels using the provided dot_coords and dot_labels.
        point_labels_dict = {tuple(map(int, point)): label for point, label in zip(dot_coords, dot_labels)}
        superpixel_labels = self._assign_labels_to_superpixels(superpixels, point_labels_dict)

        # Step 2: Apply the selected label propagation model to the superpixels.
        if self.label_propagation_model == 'DIFFUSION':
            superpixel_labels = self._diffusion_label_propagation(superpixels, superpixel_labels, img)
        elif self.label_propagation_model == 'GRAPH':
            superpixel_labels = self._graph_label_propagation(superpixels, superpixel_labels, img)
        elif self.label_propagation_model == 'DISTANCE':
            superpixel_labels = self._by_distance_label_propagation(superpixels, superpixel_labels, img)
        elif self.label_propagation_model == 'RANDOM_WALK':
            superpixel_labels = self._random_walk_label_propagation(superpixels, superpixel_labels, img)
        elif self.label_propagation_model == 'KNN':
            superpixel_labels = self._knn_labels_propagation(superpixels, superpixel_labels, img)
        elif self.label_propagation_model == 'MEAN_SHIFT':
            superpixel_labels = self._mean_shift_labels_propagation(superpixels, superpixel_labels, img)
        elif self.label_propagation_model == 'SPECTRAL':
            superpixel_labels = self._spectral_labels_propagation(superpixels, superpixel_labels, img)
        else:
            raise ValueError(f"Unknown label propagation model: {self.label_propagation_model}")

        return superpixel_labels


    # SUPERPIXEL MODELS AND METHODS


    def _slic_superpixels(self, img):
        return slic(img, n_segments=200, compactness=10)

    def _quickshift_superpixels(self, img, kernel_size=3, max_dist=6, ratio=0.5):
        return quickshift(img, kernel_size=kernel_size, max_dist=max_dist, ratio=ratio)

    def _felzenszwalb_superpixels(self, img):
        return felzenszwalb(img, scale=100, sigma=0.5, min_size=50)

    def _watershed_superpixels(self, img):

        # Convert image to grayscale
        gray_image = rgb2gray(img)

        # Compute gradient using Sobel
        gradient = ndi.sobel(gray_image)

        # Convert grayscale image to int for labels
        gray_image_int = img_as_ubyte(gray_image)

        # Find local maxima as markers using the integer grayscale image
        local_max_coords = peak_local_max(-gradient, min_distance=2, labels=gray_image_int)

        # Create markers array
        markers = np.zeros_like(gradient, dtype=int)
        markers[tuple(local_max_coords.T)] = np.arange(1, len(local_max_coords) + 1)

        return watershed(gradient, markers)

    def _entropy_superpixels(self, img, radius=5):

        # Convert the image to uint8 for rank filters
        gray_image = img_as_ubyte(rgb2gray(img))

        # Compute entropy
        entropy_img = entropy(gray_image, disk(radius))

        # Find peak local maxima coordinates
        local_max_coords = peak_local_max(entropy_img, min_distance=2)

        # Create a mask from the coordinates
        markers = np.zeros_like(entropy_img, dtype=int)
        markers[tuple(local_max_coords.T)] = np.arange(1, len(local_max_coords) + 1)

        return watershed(entropy_img, markers)


    # LABEL PROPAGATION METHODS AND LABELS

    def _by_distance_label_propagation(self, superpixels, superpixel_labels, image):

        centroids = np.array([np.mean(np.argwhere(superpixels == i), axis=0) for i in np.unique(superpixels)])
        distances = np.linalg.norm(centroids[:, np.newaxis] - centroids, axis=2)

        # Set threshold based on distance
        threshold = np.mean(distances)
        rag = graph.rag_mean_color(image, superpixels, mode='distance')
        propagated_labels = graph.cut_threshold(superpixels, rag, threshold)

        return propagated_labels

    def _random_walk_label_propagation(self, superpixels, superpixel_labels, image):

        # Convert image to grayscale to match the label shape
        gray_image = rgb2gray(image)

        # Mark known labels and set unknown labels to -1
        markers = np.copy(superpixel_labels)
        markers[superpixel_labels == 0] = -1  # Treat unknown labels as -1

        # Perform random walker segmentation
        propagated_labels = random_walker(gray_image, markers, beta=10)

        return propagated_labels

    def _knn_labels_propagation(self, superpixels, superpixel_labels, image):

        # Calculate centroids for each superpixel
        centroids = np.array([np.mean(np.argwhere(superpixels == i), axis=0) for i in np.unique(superpixels)])

        # Get unique superpixel indices and labels
        unique_superpixels = np.unique(superpixels)
        superpixel_label_map = np.array([superpixel_labels[superpixels == i][0] for i in unique_superpixels])

        # Separate known and unknown labels
        known_labels = unique_superpixels[superpixel_label_map > 0]
        known_centroids = centroids[superpixel_label_map > 0]
        unknown_labels = unique_superpixels[superpixel_label_map == 0]
        unknown_centroids = centroids[superpixel_label_map == 0]

        # Apply KNN classification to propagate labels to unknown superpixels
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(known_centroids, superpixel_label_map[superpixel_label_map > 0])
        predicted_labels = knn.predict(unknown_centroids)

        # Update superpixel_labels with the propagated labels
        for i, label in zip(unknown_labels, predicted_labels):
            superpixel_labels[superpixels == i] = label

        return superpixel_labels

    def _mean_shift_labels_propagation(self, superpixels, superpixel_labels, image):

        # Calculate centroids for each superpixel
        centroids = np.array([np.mean(np.argwhere(superpixels == i), axis=0) for i in np.unique(superpixels)])

        # Apply Mean Shift clustering
        mean_shift = MeanShift(bandwidth=20)
        clusters = mean_shift.fit_predict(centroids)

        # Map the clusters back to the superpixels
        for i, cluster in enumerate(clusters):
            mask = superpixels == np.unique(superpixels)[i]
            superpixel_labels[mask] = np.max(superpixel_labels[mask])  # Use majority label within the cluster

        return superpixel_labels

    def _spectral_labels_propagation(self, superpixels, superpixel_labels, image):

        # Calculate centroids for each superpixel
        centroids = np.array([np.mean(np.argwhere(superpixels == i), axis=0) for i in np.unique(superpixels)])

        # Perform spectral clustering based on centroids
        num_clusters = len(np.unique(superpixel_labels))
        spectral = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors')
        clusters = spectral.fit_predict(centroids)

        # Assign labels to each superpixel based on clusters
        for i, cluster in enumerate(clusters):
            mask = superpixels == np.unique(superpixels)[i]
            # Set the label of the entire superpixel to the majority label in that cluster
            superpixel_labels[mask] = np.max(superpixel_labels[mask])

        return superpixel_labels


    def _diffusion_label_propagation(self, superpixels, superpixel_labels, image, n_iterations=100):
        # Step 1: Create a feature matrix (e.g., mean color) for each superpixel
        num_superpixels = superpixels.max() + 1
        superpixel_features = np.zeros((num_superpixels, 3))  # 3 for RGB

        # Compute mean color features for each superpixel
        for sp in range(num_superpixels):
            mask = (superpixels == sp)

            # Check if the superpixel contains pixels
            if np.any(mask):
                superpixel_features[sp, :] = np.mean(image[mask], axis=0)
            else:
                superpixel_features[sp, :] = np.array([0, 0, 0])  # Assign a default value for empty superpixels

        # Step 2: Create an affinity matrix (using cosine similarity for color-based propagation)
        affinity_matrix = cosine_similarity(superpixel_features)

        # Step 3: Initialize known superpixel labels from sparse labels
        propagated_labels = np.full(num_superpixels, -1)
        for sp in range(num_superpixels):
            mask = (superpixels == sp)
            unique_labels = np.unique(superpixel_labels[mask])
            unique_labels = unique_labels[unique_labels != -1]
            if len(unique_labels) > 0:
                propagated_labels[sp] = unique_labels[0]  # Assign the first found label

        # Step 4: Propagate labels using diffusion-based approach
        for iteration in range(n_iterations):
            new_labels = propagated_labels.copy()
            for sp in range(num_superpixels):
                if propagated_labels[sp] == -1:  # Only propagate to unlabeled superpixels
                    neighbors = np.argsort(-affinity_matrix[sp])[:5]  # Top 5 neighbors
                    neighbor_labels = propagated_labels[neighbors]
                    unique, counts = np.unique(neighbor_labels[neighbor_labels != -1], return_counts=True)
                    if len(unique) > 0:
                        new_labels[sp] = unique[np.argmax(counts)]  # Assign most common neighbor label
            if np.array_equal(propagated_labels, new_labels):
                break  # Stop if labels stop changing
            propagated_labels = new_labels

        # Step 5: Map the superpixel-based labels back to pixel space
        dense_segmentation = np.zeros_like(superpixels)
        for sp in range(num_superpixels):
            dense_segmentation[superpixels == sp] = propagated_labels[sp]

        return dense_segmentation

    def _graph_label_propagation(self, superpixels, superpixel_labels, image):
        rag = graph.rag_mean_color(image, superpixels, mode='similarity')
        propagated_labels = graph.cut_threshold(superpixels, rag, 10)

        # Assigning propagated labels to regions without labels
        for region in np.unique(superpixels):
            if np.sum(superpixel_labels[superpixels == region]) == 0:
                superpixel_labels[superpixels == region] = propagated_labels[superpixels == region]

        return superpixel_labels


    # HELP FUNCTIONS


    def _assign_labels_to_superpixels(self, superpixels, labels):
        superpixel_labels = np.full(superpixels.shape, -1)
        for coord, label in labels.items():
            superpixel = superpixels[coord[1], coord[0]]
            superpixel_labels[superpixels == superpixel] = label
        return superpixel_labels


    # REPRESENTATION METHODS ADJUSTMENTS


    def __repr__(self) -> str:
        return f"PLASPIX(superpix_model={self.superpix_model}, label_propagation_model={self.label_propagation_model})"

    def __str__(self) -> str:
        return f"""PLASPIX:\n\tThe used superpixel model: '{self.superpix_model}' \n\tThe used label propagation model: '{self.label_propagation_model}'"""