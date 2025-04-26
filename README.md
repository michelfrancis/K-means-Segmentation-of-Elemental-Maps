# Author: Michel Francis
# Master in Biology - Bioinformatics Specialization
# Eötvös Loránd University, Budapest, Hungary
# E-mail: michelfrancis97@gmail.com

# K-means-Segmentation-of-Elemental-Maps

The code is intended for use in a research context, specifically for analyzing synchrotron measurement files.
It performs K-means segmentation using Lloyd's algorithm on a dataset of elemental maps, stored in an HDF5 file, and visualizes the results.

The code allows for segmentation with two approaches:
1. One layer of elements: The user can specify a list of elements that will be used for segmentation.
2. Two layers of elements: The user can specify two different lists of elements, where the first layer is used to mask non-interesting areas in the dataset.
The first layer is used to identify interesting areas, and the second layer is used to perform segmentation on those areas.

The code also provides options for dividing the dataset into blocks or sub-blocks for segmentation.
The code will also save the segmentation results as images in the current working directory.

The user is responsible for ensuring that the code is suitable for their specific use case and data.
