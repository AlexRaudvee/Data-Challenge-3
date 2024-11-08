# imports 
import os 
import cv2
import albumentations
import numpy as np
from PIL import Image                                         

class DataLoader:
    """
    A custom data loader for loading images and masks from a specified folder, performing sampling, 
    resizing transformations, and color mask creation for semantic segmentation tasks.

    This class handles image and mask file loading from subdirectories, provides functionality to 
    apply transformations, and sample points from masks for use in segmentation tasks.

    Parameters
    ----------
    data_folder : str
        The root directory containing image and mask data organized in subfolders.
    dataset_len_limit : int, optional
        The maximum number of samples to load from the dataset, by default None (no limit).
    sampling_size : int, optional
        The number of points to sample from each mask, by default 100.

    Attributes
    ----------
    data_folder : str
        Path to the dataset folder.
    sampling_size : int
        Number of points to sample from each mask.
    img_paths : list of str
        List of paths to all images in the dataset.
    mask_paths : list of str
        List of paths to all corresponding masks in the dataset.
    
    Raises
    ------
    FileNotFoundError
        If the specified data folder or any of its subdirectories cannot be found.
    ValueError
        If the dataset contains no valid image or mask files.
    
    Examples
    --------
    >>> data_loader = DataLoader(data_folder="data/coral", dataset_len_limit=500, sampling_size=100)
    >>> img, mask, dot_coords, dot_labels = data_loader[0]
    """

    def __init__(
        self,
        data_folder,
        dataset_len_limit=None,
        sampling_size=100,
    ):
        self.data_folder = data_folder
        self.sampling_size = sampling_size
        self.img_paths = []
        self.mask_paths = []

        for sub_folder in os.listdir(self.data_folder):
            for img_path in os.listdir(os.path.join(self.data_folder, sub_folder, 'images')):
                self.img_paths.append(os.path.join(self.data_folder, sub_folder, 'images', img_path))
                self.mask_paths.append(os.path.join(self.data_folder, sub_folder, 'masks_stitched', img_path.replace('.JPG', '_mask.png')))

        if dataset_len_limit is not None:
            self.img_paths = self.img_paths[:dataset_len_limit]
            self.mask_paths = self.mask_paths[:dataset_len_limit]

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns
        -------
        int
            Number of samples in the dataset.
        
        Examples
        --------
        >>> data_loader = DataLoader(data_folder="data/coral")
        >>> len(data_loader)
        1000  # Example output
        """
        return len(self.img_paths)

    def __getitem__(self, idx):
        """
        Returns the image, mask, keypoints, and point labels for the given index.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple
            A tuple containing the image, mask, keypoints, and point labels.

        Raises
        ------
        IndexError
            If the index is out of range.

        Examples
        --------
        >>> data_loader = DataLoader(data_folder="data/coral", sampling_size=50)
        >>> img, mask, dot_coords, dot_labels = data_loader[0]
        """
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        img = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path))
        mask = DataLoader.create_color_mask(mask)
        dot_coords, dot_labels = DataLoader.sample_points(mask, self.sampling_size)
        return img, mask, dot_coords, dot_labels

    @staticmethod
    def get_resize_transform(image_size=1024):
        """
        Creates a resizing transformation for images and masks to a specified size.

        Parameters
        ----------
        image_size : int, optional
            Desired output size of the image and mask. Defaults to 1024.

        Returns
        -------
        albumentations.Compose
            A composed transformation that resizes images and masks.

        Examples
        --------
        >>> transform = DataLoader.get_resize_transform(image_size=512)
        >>> transformed = transform(image=img, mask=mask)
        >>> resized_img, resized_mask = transformed["image"], transformed["mask"]
        """
        resize_transform = albumentations.Compose(
            [
                albumentations.Resize(
                    height=image_size,
                    width=image_size,
                    interpolation=cv2.INTER_AREA,
                    p=1,
                ),
            ]
        )
        return resize_transform

    @staticmethod
    def sample_points(mask, num_points):
        """
        Samples a specified number of random points from the mask, along with their labels.

        Parameters
        ----------
        mask : np.ndarray
            Binary or multi-class mask from which points are sampled.
        num_points : int
            Number of points to sample from the mask.

        Returns
        -------
        tuple of np.ndarray
            Tuple of (coordinates, labels) for the sampled points:
            - coordinates (np.ndarray): Array of shape (num_points, 2) with (x, y) coordinates.
            - labels (np.ndarray): Array of shape (num_points,) with label values at each sampled point.

        Examples
        --------
        >>> mask = np.zeros((1024, 1024), dtype=int)
        >>> dot_coords, dot_labels = DataLoader.sample_points(mask, num_points=100)
        """
        height, width = mask.shape
        y_coords = np.random.randint(0, height, num_points)
        x_coords = np.random.randint(0, width, num_points)
        dot_coords = np.stack((x_coords, y_coords), axis=-1)
        dot_labels = mask[y_coords, x_coords]

        return dot_coords, dot_labels

    @staticmethod
    def create_color_mask(image):
        """
        Creates a binary or multi-class mask from a color-coded RGB image.

        Parameters
        ----------
        image : np.ndarray
            Input RGB image where specific colors represent distinct classes.

        Returns
        -------
        np.ndarray
            A single-channel mask where pixel values indicate different classes.

        Examples
        --------
        >>> image = np.zeros((1024, 1024, 3), dtype=int)
        >>> mask = DataLoader.create_color_mask(image)
        """
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=int)
        red_mask = (image[:, :, 0] == 255) & (image[:, :, 1] == 0) & (image[:, :, 2] == 0)
        blue_mask = (image[:, :, 0] == 0) & (image[:, :, 1] == 0) & (image[:, :, 2] == 255)

        mask[red_mask] = 2
        mask[blue_mask] = 1

        return mask