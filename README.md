# SAM and PLASPIX development

## Description
During this project, we conducted experiments and research on various approaches for creating dense segmentation masks for corals using images of coral reefs and sparse point labels. The project focused on implementing the SAM (Segment Anything Model) and PLASPIX approaches in combination with different label propagation techniques and superpixel models to improve segmentation accuracy and efficiency.

## Badges
![GitHub License](https://img.shields.io/github/license/AlexRaudvee/Data-Challenge-3)
![GitHub Issues](https://img.shields.io/github/issues/AlexRaudvee/Data-Challenge-3)
![GitHub Forks](https://img.shields.io/github/forks/AlexRaudvee/Data-Challenge-3)
![GitHub Stars](https://img.shields.io/github/stars/AlexRaudvee/Data-Challenge-3)

## Visuals
To better understand the results and the models used in this project, here are some visual aids:

#### Original data and Super Pixels (SLIC Example)
![Original Image with Sparse Point Labels](assets/vis/base_example.png) ![Super Pixels (SLIC example)](assets/vis/super_pix_example.png)
#### Propagated Lables (Diffusion)
![Super Pixels (SLIC example)](assets/vis/super_pix_example.png)

#### Coral Segmentation Examples with SAM:
![Coral Segmentation](assets/vis/soft-hard-vis.png)

#### Coral Segmentation Examples with PLASPIX (SLIC + Diffusion):
- ![PLASPIX Example](assets/vis/soft-hard-vis-plaspix.png)


## Installation
To set up and run this project locally, follow these steps:

#### Requirements
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
4. Installhthe required dependencies:
    ```bash
    pip install -r requirements.txt
    ```



## Usage
After the installation, you can run the main pipeline thorugh your google colab session. The file that you have to run is `pipeline.ipynb`. Make sure to ckeck that you have created a short cut for images that are on google drive or/and change the global paths.

## Contributing
We welcome contributions! To contribute:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/new-feature`).
3. Commit your changes (`git commit -m 'Add a new feature'`). 
4. Push to the branch (`git push origin feature/new-feature`).
5. Open a Pull Request.

Make sure to check our contributing guidelines for more details.

## Authors and acknowledgment
This project was created by `Aleksandr Raudvee`, `Ansant Omurzakov`, `Leo Yung`, `Cris Bitca`, `Timofey Iukhnov`, `Igor Freik`. Special thanks to all the contributors and the open-source community for providing valuable tools and libraries that made this project possible.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Project status
Currently, the project is under active development. Future plans include adding new features and improving model performance. If you'd like to contribute or take over certain aspects, feel free to contact us!
