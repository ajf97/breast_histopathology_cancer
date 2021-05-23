<h1 align="center">Deep Learning techniques for breast cancer detection 👋</h1>
<p>
  <img alt="Version" src="https://img.shields.io/badge/version-v1.0.0-blue.svg?cacheSeconds=2592000" />
    <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/ajf97/
breast_histopathology_cancer">
  <a href="https://ajf97.github.io/blog/assets/pdf/T%C3%A9cnicas_de_deep_learning_para_diagn%C3%B3stico_de_c%C3%A1ncer_de_mama.pdf" target="_blank">
    <img alt="Documentation" src="https://img.shields.io/badge/documentation-yes-brightgreen.svg" />
  </a>
  <a href="https://opensource.org/licenses/MIT" target="_blank">
    <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg" />
  </a>
</p>

> This repository contains the code for generate classification and segmentation models of my final degree project

### 🏠 [Homepage]()

## Usage

### IDC classification

Spyder is recommended for run code. Spyder is included by default in the Anaconda Python distribution.

1. Download [dataset](https://kaggle.com/paultimothymooney/breast-histopathology-images).
2. Modify `config/breast_histopathology_cancer_config.py` and set dataset path.
3. Run `delete_noisy_images.py` for delete noisy images.
4. Run `build_breast_histopathology_cancer.py`. Output files will be in HDF5 extension.
5. If you want to reproduce all experiments, run `scripts` in `models` folder (`train_*.py` files).
6. Models will be available in `output` folder.
7. Modify and run `test_model.py` for evaluate models.

### Cell nuclei segmentation

1. Download [dataset](https://zenodo.org/record/2579118#.X6VyvCyg-iP).
2. Run script `build_tnbc.py` for generate `.npy` files.
3. Upload `.npy` files to Google Colab.
4. Open `notebooks/Segmentación semántica.iypnb` in Google Colab. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ajf97/breast_histopathology_cancer/blob/master/notebooks/Segmentación%20semántica.ipynb)
5. Models will save in your selected folder of Google Drive.

## Examples

Next picture shows IDC classification using VGG16 architecture:

<p align="center">
  <img src="images/idc_classification.png">
</p>

If you run the notebook for nuclei segmentation problem, you will obtain a mask comparison of test dataset:

<p align="center">
  <img width="500" height="" src="images/nuclei_segmentation.png">
</p>

## Author

👤 **Alejandro Jerónimo Fuentes**

- Website: https://ajf97.github.io/blog
- Github: [@ajf97](https://github.com/ajf97)
- LinkedIn: [@ajf97](https://linkedin.com/in/ajf97)

## 🤝 Contributing

Contributions, issues and feature requests are welcome!<br />Feel free to check [issues page](https://github.com/ajf97/breast_histopathology_cancer/issues).

## Show your support

Give a ⭐️ if this project helped you!

## 📝 License

Copyright © 2021 [Alejandro Jerónimo Fuentes](https://github.com/ajf97).<br />
This project is [MIT](https://opensource.org/licenses/MIT) licensed.

---

_This README was generated with ❤️ by [readme-md-generator](https://github.com/kefranabg/readme-md-generator)_
