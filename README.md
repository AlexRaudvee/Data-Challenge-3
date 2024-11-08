# SAM and PLASPIX development

## Description
During this project, we conducted experiments and research on various approaches for creating dense segmentation masks for corals using images of coral reefs and sparse point labels. The project focused on implementing the SAM (Segment Anything Model) and PLASPIX approaches in combination with different label propagation techniques and superpixel models to improve segmentation accuracy and efficiency. 

## Inspiration and Data
The initial inspiration was from one of the startups [Fruitpuch AI Team and one of their challenges](https://app.fruitpunch.ai/challenge/ai-for-coral-reefs-2#overview). All the data that were used can be found [here](http://data.qld.edu.au/public/Q1281/) and in our case we used photo quadrants.

## Table of Contents
- [Description](#description)
- [Inspiration and Data](#inspiration-and-data)
- [Badges](#badges)
- [Visuals](#visuals)
  - [Original Data -\> Superpixels -> Propagation](#original-data---super-pixels-slic---diffusion-label-propagation)
  - [Coral Segmentation Examples with SAM and PLASPIX](#coral-segmentation-examples-with-sam--plaspix-slicdiffusion)
- [Installation](#installation) - careful with python version if you run locally 
  - [Requirements](#requirements)
  - [Steps](#steps)
- [Project Derictory Structure](#project-directory-structure)
- [Usage](#usage) - Note on Google Colab Run!!!
- [Contributing](#contributing)
- [Authors and Acknowledgment](#authors-and-acknowledgment)
- [License](#license)
- [Project Status](#project-status)


## Badges
![GitHub License](https://img.shields.io/github/license/AlexRaudvee/Data-Challenge-3)
![GitHub Issues](https://img.shields.io/github/issues/AlexRaudvee/Data-Challenge-3)
![GitHub Forks](https://img.shields.io/github/forks/AlexRaudvee/Data-Challenge-3)
![GitHub Stars](https://img.shields.io/github/stars/AlexRaudvee/Data-Challenge-3)

## Visuals
To better understand the results and the models used in this project, here are some visual aids:

#### Original data -> Super Pixels (SLIC) -> Diffusion Label propagation 

<div style="display: flex; align-items: center;">
  <img src="assets/vis/base_example.png" alt="Original Image with Sparse Point Labels" width="33%" style="margin-right: 10px;">
  <img src="assets/vis/super_pix_example.png" alt="Super Pixels (SLIC example)" width="28.25%">
  <img src="assets/vis/label_propagarion_example.png" alt="Super Pixels (SLIC example)" width="31.3%">
</div>

#### Coral Segmentation Examples with SAM | PLASPIX (SLIC+Diffusion)
<div style="display: flex; align-items: center;">
  <img src="assets/vis/soft-hard-vis.png" alt="SAM Examples" width="50%" style="margin-right: 10px;">
  <img src="assets/vis/soft-hard-vis-plaspix.png" alt="PLASPIX Examples" width="50%">
</div>


## Installation
To set up and run this project locally, follow these steps:

#### Requirements
- `Python - 3.10.14` !
All the requirements can be found under `requirements.txt` file. If you would like to run the notebook locally, make sure to adjust the file paths in the `pipeline.ipynb` and remove the `google` imports in the Imports section. 

#### Steps:
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/yourproject.git
    ```
2. Navigate to the project directory:
    ```bash
    cd yourproject
    ```
3. Create and activate a virtual environment (optional but recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
4. Installable required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
5. **If you find problems with *pydencrf* library**:    
    If the problem originates from the wheel, look up the wheel online and download it manually 
    If the problem originates somewhere else, check the python version

## Project Directory Structure

```plaintext
root
├── assets/                                  # Here is all the visuals
├── exp_notebooks/
│   ├── plaspix_exp.ipynb                    # Experimentation with PLASPIX
│   └── propagators_exp.ipynb                # Experimentation with label propagation methods
├── packages/
│   ├── labelmate/                           # LabelMate package for labeling utilities FROM PREVIOUS WORK
│   ├── plaspix/                             # PLASPIX package for segmentation FROM PREVIOUS WORK
│   ├── dataloader.py                        # Custom Data loading utilities for PLASPIX and SAM 
│   ├── PLASPIX.py                           # Custom Core PLASPIX algorithm implementation
│   └── utils.py                             # Custom Utility functions for PLASPIX and SAM
├── prev_work_notebooks/                     # Previous work and exploration notebooks FROM PREVIOUS WORK
│   ├── eda-label-mismatch.ipynb             # Exploratory analysis on label mismatches
│   ├── eda-seaview.ipynb                    # Exploratory analysis on sea view data
│   ├── label-propagation-hyperparam.ipynb   # Hyperparameter tuning for label propagation
│   ├── plaspix-hyper-param-tuning.ipynb     # Hyperparameter tuning for PLASPIX
│   ├── sam-exploration.ipynb                # SAM model exploration notebook
│   └── sam-exploration-params.ipynb         # SAM exploration with parameter tuning
├── processings/
│   ├── pre_post_process.py                  # Custom Pre- and post-processing functions
├── .gitignore                               # Git ignore file
├── LICENSE                                  # License for the project
├── README.md                                # Project README file
├── requirements.txt                         # Python dependencies
└── pipeline.ipynb                           # Main pipeline notebook
```

## Usage
After the installation, you can run the main pipeline through your google colab session. The file that you have to run is `pipeline.ipynb`. Make sure to check that you have created a short cut for images that are on google drive or/and change the global paths. 
**NOTE**: If you run the pipeline in google colab, make sure to add the following folders to working environment: `packages`, `processings`, `assets`.

## Contributing
We welcome contributions! To contribute:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/new-feature`).
3. Commit your changes (`git commit -m 'Add a new feature'`). 
4. Push to the branch (`git push origin feature/new-feature`).
5. Open a Pull Request.

Make sure to check our contributing guidelines for more details.

## Authors and acknowledgment
This project was created by `Aleksandr Raudvee`, `Ansant Omurzakov`, `Leo Yang`, `Cris Bitca`, `Timofey Iukhnov`, `Igor Freik`. Special thanks to all the contributors and the open-source community for providing valuable tools and libraries that made this project possible.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Project status
Currently, the project is under active development. Future plans include adding new features and improving model performance. If you'd like to contribute or take over certain aspects, feel free to contact us!
