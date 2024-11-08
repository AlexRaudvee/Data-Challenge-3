# Standard library imports
import os                                                       # For handling file paths and directories

# Data science and numerical computation libraries
import numpy as np                                              # NumPy for array and matrix operations
import matplotlib.pyplot as plt                                 # Matplotlib for plotting graphs and visualizations
from matplotlib.colors import ListedColormap                    # Color maps for matplotlib plots

# Utilities for progress tracking and notebook environments
from tqdm.auto import tqdm                                      # Progress bar utility for loops (auto-adjusts to the environment)
from tqdm.notebook import tqdm                                  # Notebook-specific progress bar for Jupyter/Colab

# Typing module for type hinting in functions
from typing import Iterable, Callable                           # Type hinting utilities

# custom imports 
from PLASPIX import PLASPIX
from ..processings.pre_post_processing import *

# Segment Anything Model (SAM) for segmentation tasks
from segment_anything import SamPredictor, sam_model_registry   # SAM model predictor and registry
from sam2.build_sam import build_sam2                           # SAM2 builder
from sam2.sam2_image_predictor import SAM2ImagePredictor        # SAM2 image predictor for segmentation

MASK_CLASSES = [1, 2]


def iou_score(pred_mask: np.array, gt_mask: np.array, class_label: int):
    """
    Computes the Intersection over Union (IoU) score between the predicted mask and the ground truth mask
    for a specific class label.

    Parameters
    -----------
    pred_mask : np.array
        Predicted mask as a numpy array with values 0, 1, or 2.
    gt_mask : np.array
        Ground truth mask as a numpy array with values 0, 1, or 2.
    class_label : int
        The class label (e.g., 0, 1, or 2) for which to calculate the IoU score.

    Returns
    --------
    float
        IoU score for the given class between the predicted and ground truth masks.
    """

    pred_binary_mask = (pred_mask == class_label).astype(np.uint8)
    gt_binary_mask = (gt_mask == class_label).astype(np.uint8)

    # # Debugging: Print unique labels in both masks
    # pred_unique_labels = np.unique(pred_mask)
    # gt_unique_labels = np.unique(gt_mask)
    # print(f"Unique labels in predicted mask: {pred_unique_labels}")
    # print(f"Unique labels in ground truth mask: {gt_unique_labels}")

    intersection = np.logical_and(pred_binary_mask, gt_binary_mask).sum()
    union = np.logical_or(pred_binary_mask, gt_binary_mask).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    iou = intersection / union
    print(f"IoU for class {class_label}: {iou}")
    return iou


def pixel_accuracy(gt_mask: np.array, pred_mask: np.array):
    correct = np.sum(pred_mask == gt_mask)
    total = pred_mask.size
    accuracy = correct / total
    return accuracy


def predict_sam_with_dots(predictor, image, dot_coords, dot_labels):
    """
    Predicts a mask using the SAM (Segment Anything Model) given the image and dot coordinates as inputs.

    Parameters
    -----------
    predictor : Callable
        SAM model predictor.
    image : np.array
        Input image as a numpy array.
    dot_coords : list or np.array
        Coordinates of dots indicating points of interest on the image.
    labels : list or np.array
        Labels corresponding to each dot (1 for positive, 0 for negative).

    Returns
    --------
    np.array
        Predicted mask as a numpy array.
    """
    pred_mask = np.zeros(image.shape[:2])

    for m_class in MASK_CLASSES:
        predictor.set_image(image)
        input_points = np.array(dot_coords)
        input_labels = np.where(dot_labels == m_class, 1, 0)
        masks, _, _ = predictor.predict(point_coords=input_points, point_labels=input_labels, multimask_output=False)
        pred_mask[masks[0] == 1] = m_class
    return pred_mask


def predict_plaspix_with_dots(PLASPIX_model: PLASPIX, image, dot_coords, dot_labels) -> np.array:
    """
    Predicts labels for an image using the PLASPIX model, with label propagation based on
    user-provided dot coordinates and corresponding labels.

    This function utilizes a pre-initialized PLASPIX model to first generate superpixels
    from the input image. Then, it propagates the provided dot labels across the superpixels
    to produce a labeled output.

    Parameters
    -----------
    PLASPIX_model : PLASPIX
        An instance of the PLASPIX class, which has been initialized with a specific superpixel
        segmentation method and label propagation algorithm.

    img : ndarray
        The input image as a NumPy array (typically in RGB or grayscale format) on which
        superpixels and label propagation will be applied.

    dot_coords : list of tuples
        A list of coordinates where labels are provided. Each tuple contains two integers,
        representing the (row, column) coordinates of a point in the image where the corresponding
        label is known.

    dot_labels : list of integers
        A list of labels corresponding to the points in `dot_coords`. The i-th label in this list
        corresponds to the label of the i-th point in `dot_coords`. These labels will serve as the
        seeds for propagation over the superpixels.

    Returns
    --------
    mask : np.array
        A 2D array representing the propagated labels for each superpixel in the image. The size of
        the array matches the height and width of the input image, where each pixel is assigned the
        propagated label of its respective superpixel.

    Example
    --------
    ```
    # Assuming PLASPIX_model has been initialized with 'SLIC' for superpixels and 'KNN' for label propagation
    img = np.random.rand(300, 300, 3)  # Example image
    dot_coords = [(50, 50), (150, 200), (250, 250)]  # Example points
    dot_labels = [1, 2, 3]  # Corresponding labels for the points

    # Predict labels with the PLASPIX model
    superpixel_labels = predict_plaspix_with_dots(PLASPIX_model, img, dot_coords, dot_labels)

    # `superpixel_labels` contains the propagated labels across the image
    ```

    Notes
    ------
    - Ensure that the PLASPIX model is already initialized with appropriate algorithms for both
      superpixel generation and label propagation before calling this function.
    - The dot coordinates should match the dimensions of the input image.
    - This function performs two main steps: (1) superpixel segmentation and (2) label propagation,
      resulting in labeled superpixels across the entire image.
    """

    pred_mask = np.zeros(image.shape[:2])

    for m_class in MASK_CLASSES:
        input_points = np.array(dot_coords)
        input_labels = np.where(dot_labels == m_class, m_class, 0)
        superpixels = PLASPIX_model.generate_superpixels(image)
        superpixel_labels = PLASPIX_model.propagate_labels(superpixels, input_points, input_labels, image)

    return superpixel_labels


def predict_mask(model, model_type: str, preprocess: bool, postprocess: bool, img, dot_coords, dot_labels):
    """
    Predicts the mask for the given image using the specified model.

    Parameters
    -----------
    model : Callable
        The model to be used for prediction.
    model_type : str
        Type of the model (e.g., 'SAM').
    img : np.array
        Input image as a numpy array.
    dot_coords : list or np.array
        Coordinates of dots indicating points of interest on the image.
    dot_labels : list or np.array
        Labels corresponding to each dot (1 for positive, 0 for negative).

    Returns
    --------
    np.array
        Predicted mask as a numpy array.
    """

    if preprocess:
        img = normalize(img)

    if model_type == 'SAM':
        pred_mask = predict_sam_with_dots(model, img, dot_coords, dot_labels)
    elif model_type == 'PLASPIX':
        pred_mask = predict_plaspix_with_dots(model, img, dot_coords, dot_labels)
    else:
        pred_mask = model(img, dot_coords, dot_labels)

    if postprocess:
        pred_mask = lc_post(pred_mask)

    return pred_mask


def eval_pipeline(dataset: Iterable, model: Callable, model_type: str, preprocess: bool, postprocess: bool):
    """
    Evaluates the performance of the model on a dataset by calculating the mean IoU score.

    Parameters
    -----------
    dataset : Iterable
        Dataset containing images, masks, and point annotations.
    model : Callable
        The model to be evaluated.
    model_type : str, optional
        Type of the model (e.g., 'SAM'). Defaults to 'SAM'.

    Returns
    --------
    list
        List of IoU scores for each sample in the dataset.
    """
    ious = []
    pixel_accuracies = []

    for img, mask, dot_coords, dot_labels in tqdm(dataset):
        pred_mask = predict_mask(model, model_type, preprocess, postprocess, img, dot_coords, dot_labels)
        scores = [iou_score(pred_mask, mask, class_label=m_class) for m_class in MASK_CLASSES]
        pixel_accuracies.append(pixel_accuracy(mask, pred_mask))
        mean_score = sum(scores) / len(scores)
        ious.append(mean_score)

    print(f'mIoU score for {model_type}:{model}: {sum(ious) / len(ious)}')
    print(f'Pixel Accuracy for {model_type}:{model}: {sum(pixel_accuracies) / len(pixel_accuracies)}')
    return ious, pixel_accuracies


def show_points(coords, labels, ax, binary=True, marker_size=50):
    """
    Visualizes points on the image, showing positive and negative points in different colors.

    Parameters
    -----------
    coords : np.array
        Coordinates of points to be visualized.
    labels : np.array
        Labels for each point (1 for positive, 0 for negative).
    ax : matplotlib.axes.Axes
        The axis on which the points are to be plotted.
    marker_size : int, optional
        Size of the marker for each point. Defaults to 50.
    """
    if binary:
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', s=marker_size, edgecolor='white', linewidth=1.25)
    else:
        cmap = ListedColormap(['black', 'blue', 'red'])
        ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap=cmap, s=marker_size, edgecolor='white', linewidth=1.25)


def visualize(dataset, model, model_type, preprocess: bool, postprocess: bool, num_images=5):
    """
    Visualizes the predictions of the model alongside ground truth for a subset of images.

    Parameters
    -----------
    model : Callable
        The model used for prediction.
    model_type : str
        Type of the model (e.g., 'SAM').
    dataset : Iterable
        Dataset containing images, masks, and point annotations.
    num_images : int, optional
        Number of images to visualize. Defaults to 5.
    """
    fig, axs = plt.subplots(num_images, 3, figsize=(10, num_images * 3))

    # If num_images is 1, axs will not be a 2D array, so we adjust accordingly
    if num_images == 1:
        axs = [axs]  # Make it a list so we can index as if it were 2D

    for i in range(num_images):
        image, gt_mask, points, point_labels = dataset[i]
        predicted_mask = predict_mask(model, model_type, preprocess, postprocess, image, points, point_labels)

        axs[i][0].imshow(image)  # Access axs in the 1D case
        axs[i][0].set_title("Input Image")
        axs[i][0].axis('off')
        # show_points(points, np.vectorize(LABEL2BIT.get)(point_labels), axs[i, 0])
        show_points(points, point_labels, axs[i][0], binary=False)

        # Define a color map: 0 -> black, 1 -> blue, 2 -> red
        cmap = ListedColormap(['black', 'blue', 'red'])
        axs[i][1].imshow(gt_mask, cmap=cmap)
        axs[i][1].set_title("Ground Truth Mask")
        axs[i][1].axis('off')

        axs[i][2].imshow(predicted_mask, cmap=cmap)
        axs[i][2].set_title("Predicted Mask")
        axs[i][2].axis('off')

        print(f"{i} out of {num_images}")

    plt.tight_layout()
    plt.show()


def predict_pipeline(dataset: Iterable, model: Callable, model_type: str, preprocess: bool, postprocess: bool, output_dir: str):
    """
    Runs the prediction pipeline on a dataset and saves the predicted masks to a specified directory.

    Parameters
    ----------
    dataset : Iterable
        Dataset containing tuples of images, masks, and point annotations.
    model : Callable
        The model function used to make predictions, compatible with the specified `model_type`.
    model_type : str
        Identifier for the type of model being used, e.g., 'SAM'.
    preprocess : bool
        Whether to apply preprocessing to each image before prediction.
    postprocess : bool
        Whether to apply postprocessing to the predicted masks.
    output_dir : str
        Directory where predicted masks should be saved as NumPy arrays.

    Returns
    -------
    None
        Saves each predicted mask as a `.npy` file in the `output_dir`.

    Raises
    ------
    FileNotFoundError
        If `output_dir` cannot be created due to invalid path.
    ValueError
        If `model_type` is unsupported or incompatible with the `model` function.

    Examples
    --------
    >>> dataset = CustomDataset()
    >>> model = load_model("SAM")
    >>> predict_pipeline(dataset, model, model_type="SAM", preprocess=True, postprocess=True, output_dir="predictions")
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, (img, _, dot_coords, dot_labels) in enumerate(tqdm(dataset)):
        # Predict the mask
        pred_mask = predict_mask(model, model_type, preprocess, postprocess, img, dot_coords, dot_labels)

        # Save the predicted mask as a NumPy array
        np.save(os.path.join(output_dir, f"pred_mask_{i}.npy"), pred_mask)

    print(f"Predicted masks saved to {output_dir}")


def eval_pipeline(dataset: Iterable, output_dir: str):
    """
    Evaluates model performance by comparing predicted masks with ground truth masks.

    Parameters
    ----------
    dataset : Iterable
        Dataset containing tuples of images, ground truth masks, and point annotations.
    output_dir : str
        Directory where the predicted masks are saved as NumPy arrays.

    Returns
    -------
    ious : List[float]
        List of Intersection over Union (IoU) scores for each sample in the dataset.
    pixel_accuracies : List[float]
        List of pixel accuracy scores for each sample in the dataset.

    Raises
    ------
    FileNotFoundError
        If predicted mask files cannot be found in `output_dir`.
    ValueError
        If `dataset` and `output_dir` do not contain the same number of samples.

    Examples
    --------
    >>> dataset = CustomDataset()
    >>> output_dir = "predictions"
    >>> ious, pixel_accuracies = eval_pipeline(dataset, output_dir)
    >>> print("IoU scores:", ious)
    >>> print("Pixel accuracies:", pixel_accuracies)
    """
    ious = []
    pixel_accuracies = []

    for i, (_, gt_mask, _, _) in enumerate(tqdm(dataset)):
        # Load the predicted mask from the output directory
        pred_mask_path = os.path.join(output_dir, f"pred_mask_{i}.npy")
        pred_mask = np.load(pred_mask_path)

        # Calculate IoU score for each class
        scores = [iou_score(pred_mask, gt_mask, class_label=m_class) for m_class in MASK_CLASSES]

        # Calculate pixel accuracy
        pixel_acc = pixel_accuracy(gt_mask, pred_mask)

        # Append results
        mean_iou = sum(scores) / len(scores)
        ious.append(mean_iou)
        pixel_accuracies.append(pixel_acc)

    # Compute and print overall performance metrics
    mean_iou_score = sum(ious) / len(ious)
    mean_pixel_accuracy = sum(pixel_accuracies) / len(pixel_accuracies)

    print(f"Mean IoU Score: {mean_iou_score}")
    print(f"Mean Pixel Accuracy: {mean_pixel_accuracy}")

    return ious, pixel_accuracies


def visualize(dataset, output_dir: str, num_images=5):
    """
    Visualizes model predictions alongside ground truth images and masks.

    Parameters
    ----------
    dataset : Iterable
        A dataset containing tuples with images, ground truth masks, and point annotations.
    output_dir : str
        Directory containing saved predicted masks as NumPy array files.
    num_images : int, optional
        Number of image samples to visualize. Defaults to 5.

    Returns
    -------
    None
        Displays a plot with columns showing the input image, ground truth mask, and predicted mask.

    Raises
    ------
    FileNotFoundError
        If a specified predicted mask file does not exist in `output_dir`.
    ValueError
        If `num_images` exceeds the number of samples in the dataset.

    Examples
    --------
    >>> dataset = CustomDataset()  # Assume this is a dataset with image and mask data
    >>> output_dir = "predictions"
    >>> visualize(dataset, output_dir, num_images=3)
    """
    fig, axs = plt.subplots(num_images, 3, figsize=(10, num_images * 3))

    # Adjust subplot handling if num_images is 1
    if num_images == 1:
        axs = [axs]

    for i in range(num_images):
        image, gt_mask, points, point_labels = dataset[i]
        pred_mask_path = os.path.join(output_dir, f"pred_mask_{i}.npy")

        try:
            predicted_mask = np.load(pred_mask_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Predicted mask file not found: {pred_mask_path}")

        # Plot input image
        axs[i][0].imshow(image)
        axs[i][0].set_title("Input Image")
        axs[i][0].axis('off')
        show_points(points, point_labels, axs[i][0], binary=False)

        # Define a color map: 0 -> black, 1 -> blue, 2 -> red
        cmap = ListedColormap(['black', 'blue', 'red'])

        # Plot ground truth mask
        axs[i][1].imshow(gt_mask, cmap=cmap)
        axs[i][1].set_title("Ground Truth Mask")
        axs[i][1].axis('off')

        # Plot predicted mask
        axs[i][2].imshow(predicted_mask, cmap=cmap)
        axs[i][2].set_title("Predicted Mask")
        axs[i][2].axis('off')

        print(f"{i + 1} out of {num_images}")

    plt.tight_layout()
    plt.show()


def load_sam_model(model_type="vit_h", checkpoint="../checkpoints/sam_vit_h_4b8939.pth"):
    """
    Loads a SAM (Segment Anything Model) model from a checkpoint file.

    Parameters
    ----------
    model_type : str, optional
        Type of SAM model to load, e.g., 'vit_h', 'vit_b'. Defaults to 'vit_h'.
    checkpoint : str, optional
        Path to the model checkpoint file. Defaults to "../checkpoints/sam_vit_h_4b8939.pth".

    Returns
    -------
    SamPredictor
        An instance of the SamPredictor, initialized with the specified model.

    Raises
    ------
    FileNotFoundError
        If the specified checkpoint file does not exist.
    ValueError
        If the `model_type` is unsupported.

    Examples
    --------
    >>> predictor = load_sam_model(model_type="vit_h", checkpoint="checkpoints/sam_vit_h.pth")
    >>> prediction = predictor.predict(image)
    """
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")

    model = sam_model_registry[model_type](checkpoint=checkpoint)
    model = model.to('cuda')
    return SamPredictor(model)


def load_sam2_model(model_type="sam2.1_hiera_large", model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml"):
    """
    Loads a SAM2 model from a configuration file and a checkpoint.

    Parameters
    ----------
    model_type : str, optional
        Type of SAM2 model to load, e.g., 'sam2.1_hiera_large'. Defaults to 'sam2.1_hiera_large'.
    model_cfg : str, optional
        Path to the configuration file for the SAM2 model. Defaults to "configs/sam2.1/sam2.1_hiera_l.yaml".

    Returns
    -------
    SAM2ImagePredictor
        An instance of the SAM2ImagePredictor, initialized with the SAM2 model.

    Raises
    ------
    FileNotFoundError
        If the specified configuration or checkpoint file does not exist.
    ValueError
        If `model_type` is unsupported or incompatible with `model_cfg`.

    Examples
    --------
    >>> predictor = load_sam2_model(model_type="sam2.1_hiera_large",
    ...                             model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml")
    >>> prediction = predictor.predict(image)
    """
    checkpoint = f"../checkpoints/{model_type}.pt"
    
    if not os.path.exists(model_cfg):
        raise FileNotFoundError(f"Configuration file not found: {model_cfg}")
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")

    sam2_model = build_sam2(model_cfg, checkpoint, device='cuda')
    predictor = SAM2ImagePredictor(sam2_model)
    return predictor
