# Guía de uso

## Clasificación de IDC

Para ejecutar los experimentos, se recomienda el uso de IDE Spyder, disponible en el entorno Anaconda.

1. Descargar el *dataset* desde [aquí](https://kaggle.com/paultimothymooney/breast-histopathology-images).
2. Modificar el archivo *config/breast_histopathology_cancer_config.py* según la ruta del *dataset*.
3. Ejecutar el *script* `delete_noisy_images.py` para eliminar las imágenes innecesarias del disco.
4. Ejecutar el *script* `build_breast_histopathology_cancer.py`. La salida serán varios archivos en formato HDF5 con los datos.
5. Para reproducir los experimentos, ejecutar los *scripts* de la carpeta *models* (archivos `train_*.py`).
6. Los modelos estarán disponibles en la carpeta *output*. Las gráficas se verán en Spyder.
7. Para evaluar el modelo, modificar y ejecutar el *script* `test_model.py`. Los resultados se muestran en la terminal de Spyder.

## Segmentación de células cancerosas.

1. Descargar el *dataset* desde [aquí](https://zenodo.org/record/2579118#.X6VyvCyg-iP).
2. Ejecutar el *script* `build_tnbc.py` para generar los archivos *.npy*.
3. Subir los archivos *.npy* a Google Colab.
4. Ejecutar el archivo `notebooks/Segmentación semántica.iypnb` en Google Colab.
5. Los modelos se guardarán en la carpeta seleccionada de Google Drive.