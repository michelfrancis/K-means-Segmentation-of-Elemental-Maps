
# ==================================================
# 
# Author: Michel Francis
# Master in Biology - Bioinformatics Specialization
# Eötvös Loránd University, Budapest, Hungary
# 
# E-mail: michelfrancis97@gmail.com
# GitHub: https://github.com/michelfrancis/
# 
# Last Update: 2025.04.26
# 
# ==================================================


# ==================================================
# ==================================================
# 
# K-means Segmentation of Elemental Maps
# 
# ==================================================
# ==================================================


# --------------------------------------------------
# Description:
# --------------------------------------------------

# The code is intended for use in a research context, specifically for analyzing synchrotron measurement files.
# It performs K-means segmentation using Lloyd's algorithm on a dataset of elemental maps, stored in an HDF5 file, and visualizes the results.

# The code allows for segmentation with two approaches:
# 1. One layer of elements: The user can specify a list of elements that will be used for segmentation.

# 2. Two layers of elements: The user can specify two different lists of elements, where the first layer is used to mask non-interesting areas in the dataset.
# The first layer is used to identify interesting areas, and the second layer is used to perform segmentation on those areas.

# The code also provides options for dividing the dataset into blocks or sub-blocks for segmentation.
# The code will also save the segmentation results as images in the current working directory.

# The user is responsible for ensuring that the code is suitable for their specific use case and data.


# ==================================================
#
# USER INPUTS
# 
# ==================================================


# --------------------------------------------------
# Please write the dataset file path, parameters group path, energy level, elements, number of clusters, and division level in the following lines.
# --------------------------------------------------

# FilePath: Path to the data file, it should be a .h5 file.
FilePath = "/0-Synchrotron Measurement Files/fitted_ATIS_C1_rep1_roi25950_33280.h5"

# ParametersPath: Path to the parameters group that contains the elemental maps in the HDF5 file.
ParametersPath = "/fitted_stack/stack.1/results/parameters"

# EnergyLevel: Energy level for segmentation, either "HIGH" or "LOW".
EnergyLevel = "HIGH" # "HIGH" or "LOW"

# ElementsLayer1: List of elements in the first layer, there should be at least one element in this list in a string format eg. ElementsLayer1 = ["P", "S"].
ElementsLayer1 = ["P", "S"]

# ElementsLayer2: List of elements in the second layer (if applicable).
# If you want to use only one layer, please write the elements in ElementsLayer1 and leave ElementsLayer2 empty eg. ElementsLayer2 = [].
ElementsLayer2 = ["P", "S", "Fe", "Mg"] 

# K1 and K2: Number of clusters for the first and second layers respectively.
# If you want to use only one layer, you can write any value for K2, and it will be ignored.
K1 = 3
K2 = 3

# DivisionLevel: Level of division for segmentation (0: No blocks, 1: 4 blocks, 2: 16 sub-blocks).
DivisionLevel = 0

# --------------------------------------------------
# Click on Run to execute the code, and the segmentation results will be displayed and saved.
# --------------------------------------------------


# ==================================================
# 
# IMPORT LIBRARIES
# 
# ==================================================


import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import time


# ==================================================
# 
# DEFINE FUNCTIONS
# 
# ==================================================


# --------------------------------------------------
# Open the file and extract the data
# --------------------------------------------------

def OpenFile(FilePath, ParametersPath, EnergyLevel, Element):

    # Extract the file name from the file path
    file_name = FilePath.split('/')[-1]
    file_name = file_name.split('.')[0]

    # Open the HDF5 file and access the parameters group
    try:
        with h5py.File(FilePath, "r") as f:
            parameters = f.get(ParametersPath)
            if parameters is None:
                print(f"Parameters path '{ParametersPath}' not found in the file '{file_name}'!")
                return
        
            # Get the data for the specified element and energy level
            HighEnergyElements = ("Al", "Ca", "Cl", "Cr", "Fe", "K", "Mg", "Mn", "P", "S", "Si", "Ti")
            LowEnergyElements = ("C", "Co", "Cu", "Fe", "Mn", "N", "O")

            if EnergyLevel == "HIGH":
                if Element in HighEnergyElements:
                    data = parameters.get(Element + "_K")
                else:
                    print(f"{Element} is not in the HIGH energy level list: {HighEnergyElements}!")
                    return

            elif EnergyLevel == "LOW":
                if Element in ("Co", "Cu", "Fe", "Mn"):
                    data = parameters.get(Element + "_L")
                elif Element in ("C", "N", "O"):
                    data = parameters.get(Element + "_K")
                else:
                    print(f"{Element} is not in the LOW energy level list: {LowEnergyElements}!")
                    return
            
            else:
                print(f"EnergyLevel: {EnergyLevel} is not valid. Please Choose 'HIGH' or 'LOW'!")
                return

            if data is None:
                print(f"Data for {Element} not found in the ParametersPath: '{ParametersPath}'!")
                return
            
            data = np.array(data)
            return data
    
    except:
        print(f"The file: '{FilePath}' does not exist or cannot be opened!")
        return


# --------------------------------------------------
# Flatten and Convert Data To Points
# --------------------------------------------------

def FitData(data):
    flattened_data = data.flatten()
    ElementPoints = [[item.item()] for item in flattened_data]
    return ElementPoints


# --------------------------------------------------
# Combine the Points from All Elements for K-means
# --------------------------------------------------

def PreLloyd(FilePath, ParametersPath, EnergyLevel, Elements):
    Points = []

    # Loop through each element in the provided list of Elements
    for Element in Elements:

        # Load the data for the element
        data = OpenFile(FilePath, ParametersPath, EnergyLevel, Element)
        
        # Check if the data is None (indicating an error in loading) and return None for points, height, and width
        if data is None:
            return None, None, None
        
        # Get the shape of the data to determine its height and width
        Height, Width = data.shape

        # Flatten the data and convert it to points and append to the Points list
        ElementPoints = FitData(data)
        Points.append(ElementPoints)
    
    # After processing all elements, combine the points from all elements
    Points = np.array([sum(elem, []) for elem in zip(*Points)])
    
    return Points, Height, Width


# --------------------------------------------------
# Euclidean Euclidean Distance
# --------------------------------------------------

def Distance(x, y):
    assert len(x) == len(y)
    d = 0
    for n, i in enumerate(x):
        d += (i-y[n])**2
    return d**0.5


# --------------------------------------------------
# Farthest Centers Selection
# --------------------------------------------------

def FarthestFirstTraversal(K, m, Points):
    Centers = []

    # Find the first valid point (not [nan, nan, ...]) and consider it as the first center
    for point in Points:
        if not np.isnan(point).all():
            Centers.append(point)
            break
    
    # If no valid points found, print a message and stop the function
    if len(Centers) == 0:
        print("No valid points found in the data.")
        return
    
    while len(Centers) < K:
        # Find the nearest center from the already selected centers for each valid point
        valid_points = [point for point in Points if not np.isnan(point).all()]
        NearestCenters = [min([x for x in Centers],
                               key=lambda x: Distance(i, x))
                            for i in valid_points]
        
        # Find the farthest point from the already selected centers and add it to the centers
        FarthestPoint = (max(zip(valid_points, NearestCenters), key= lambda x: Distance(*x))[0])
        Centers.append(FarthestPoint)

    return Centers


# --------------------------------------------------
# Calculate the Squared Error
# --------------------------------------------------

def SquaredErrorDistortion(m, Centers, Points):

    # Initialize the total squared error and valid points count
    TotalSquaredError = 0
    ValidPointsCount = 0

    # Iterate through each point in the dataset
    for i in Points:

        # If the point is NaN (Not a Number), skip it and go to the next point
        if np.isnan(i).all():
            continue

        # Find the nearest center to the point
        NearestCenters = min([x for x in Centers], key=lambda x: Distance(i, x))

        # Calculate the squared distance between the point and the nearest center
        SquaredDistance = Distance(i, NearestCenters) ** 2

        # Add the squared distance to the total squared error and increment the valid points count
        TotalSquaredError += SquaredDistance
        ValidPointsCount += 1

    # return the average squared error
    return TotalSquaredError/ValidPointsCount


# --------------------------------------------------
# Lloyd's Algorithm for K-means Clustering
# --------------------------------------------------

def LloydsAlgorithm(K, Height, Width, m, Points, max_iterations=1000):

    # Initialize the centers using Farthest-First Traversal and list to store distortions for each iteration
    OldCenters = FarthestFirstTraversal(K, m, Points)
    Distortions = []

    # Start the Lloyd's algorithm iterations
    for iteration in range(max_iterations):

        # Initialize a dictionary to store clusters, keys are centers and values are lists of points in that cluster
        ClustersDict = {tuple(x): [] for x in OldCenters}

        # Initialize a list to store the clustered dataset (each point will be replaced by its center value)
        ClusteredDataset = []

        # Iterate through each point in the dataset
        for i in Points:

            # If the point is NaN, append it to the ClusteredDataset and continue
            if np.isnan(i).all():
                ClusteredDataset.append(np.array(i))
                continue

            # Find the nearest center to the point by calculating the distance to each center
            MinDistance = float("inf")
            Center = None
            for x in ClustersDict.keys():
                Dis = Distance(i, x)
                if MinDistance >= Dis:
                    MinDistance = Dis
                    Center = x
            
            # Append the point to its corresponding cluster in the dictionary
            ClustersDict[Center].append(i)
            # Append the center value to the ClusteredDataset
            ClusteredDataset.append(Center)
        
        # Calculate the new centers by averaging the points in each cluster
        NewCenters = set()
        for key, value in ClustersDict.items():
            Center = []
            for i in range(m):
                DimensionsSum = sum(point[i] for point in value)
                Center.append(DimensionsSum/ len(value))
            # Append the new center to the set of new centers
            NewCenters.add(tuple(Center))
        
        # Calculate the distortion for the current iteration and append it to the list
        Distortion = SquaredErrorDistortion(m, NewCenters, Points)
        Distortions.append(Distortion)

        # Check for convergence: if the new centers are the same as the old centers, break the loop
        if NewCenters == OldCenters:
            break
        # Otherwise, update the old centers for the next iteration
        OldCenters = NewCenters

    # # Plotting the number of iterations vs. distortion
    # plt.plot(range(1, len(Distortions) + 1), Distortions)
    # plt.xlabel('Number of Iterations')
    # plt.ylabel('Squared Error Distortion')
    # plt.title('Iterations vs. Squared Error Distortion')
    # plt.show()

    # Sort the final centers based on the first element value in each center and convert them to float
    SortedCenters = sorted(OldCenters, key=lambda x: x[0])
    SortedCenters = [tuple(float(val) for val in sublist) for sublist in SortedCenters]

    # Reshape the clustered dataset into a 3D array with dimensions (Height, Width, m)
    Array3D = np.array(ClusteredDataset).reshape(Height, Width, m)

    # Extract the first element values from the 3D array (the first dimension (m=0) for each pixel in the dataset)
    FirstElementValues = Array3D[:, :, 0]

    print("Sorted Centers after Lloyd's Algorithm for the first layer:\n", SortedCenters)
    return SortedCenters, ClusteredDataset, FirstElementValues


# --------------------------------------------------
# Lloyd Layer 2 to Exclude the Non-Interesting Areas and Recalculate the Centers
# --------------------------------------------------

def LloydLayer2(SortedCenters, ClusteredDataset, K, Height, Width, m, Points, max_iterations=1000):

    # Loop through the clustered dataset and set the points corresponding to the first and last clusters to NaN
    for i in range(len(ClusteredDataset)):
        if ClusteredDataset[i] == SortedCenters[0] or ClusteredDataset[i] == SortedCenters[-1]:
            for j in range(len(Points[i])):
                (Points[i])[j] = 0
                (Points[i])[j] = np.nan
    
    # Reapply Lloyd's algorithm to the interesting areas
    SortedCenters, ClusteredDataset, FirstElementValues = LloydsAlgorithm(K, Height, Width, m, Points, max_iterations=1000)

    print("Sorted Centers after Lloyd Layer 2:\n", SortedCenters)
    return SortedCenters, ClusteredDataset, FirstElementValues


# --------------------------------------------------
# Divide the Points into Blocks - Division Level 1 (4 Blocks)
# --------------------------------------------------

def Division(Points, Height, Width, Elements):

    # Initialize empty lists to store blocks of points, their heights, and widths
    BlocksPoints, BlocksHeights, BlocksWidths =  [], [], []

    # Reshape the Points into a 3D array (Height x Width x number of elements)
    Points = np.array(Points).reshape(Height, Width, len(Elements))

    # Get the number of rows and columns from the reshaped Points array
    rows, cols = Points.shape[0], Points.shape[1]

    # Extract the top-left block (first half of rows and columns)
    top_left = Points[:rows//2, :cols//2]
    BlocksHeights.append(top_left.shape[0])
    BlocksWidths.append(top_left.shape[1])
    top_left = top_left.reshape(-1, len(Elements))
    BlocksPoints.append(top_left)

    # Extract the top-right block (first half of rows and second half of columns)
    top_right = Points[:rows//2, cols//2:]
    BlocksHeights.append(top_right.shape[0])
    BlocksWidths.append(top_right.shape[1])
    top_right = top_right.reshape(-1, len(Elements))
    BlocksPoints.append(top_right)

    # Extract bottom-left block (second half of rows and first half of columns)
    bottom_left = Points[rows//2:, :cols//2]
    BlocksHeights.append(bottom_left.shape[0])
    BlocksWidths.append(bottom_left.shape[1])
    bottom_left = bottom_left.reshape(-1, len(Elements))
    BlocksPoints.append(bottom_left)

    # Extract bottom-right block (second half of rows and second half of columns)
    bottom_right = Points[rows//2:, cols//2:]
    BlocksHeights.append(bottom_right.shape[0])
    BlocksWidths.append(bottom_right.shape[1])
    bottom_right = bottom_right.reshape(-1, len(Elements))
    BlocksPoints.append(bottom_right)

    return BlocksPoints, BlocksHeights, BlocksWidths


# --------------------------------------------------
# Divide the Points into SubBlocks - Division Level 2 (16 Blocks)
# --------------------------------------------------

def SubDivision(BlocksPoints, BlocksHeights, BlocksWidths, Elements):

    # Initialize empty lists to store subdivided points, heights, and widths for each block
    SubBlocksPoints = [[], [], [], []]
    SubBlocksHeights = [[], [], [], []]
    SubBlocksWidths = [[], [], [], []]

    # Iterate over each of the 4 blocks (top-left, top-right, bottom-left, bottom-right)
    for i in range(4):

        # Reshape the block points into a 3D array (Height x Width x number of elements)
        BlocksPoints[i] = np.array(BlocksPoints[i]).reshape(BlocksHeights[i], BlocksWidths[i], len(Elements))

        # Get the number of rows and columns from the reshaped block points array
        rows, cols = BlocksPoints[i].shape[0], BlocksPoints[i].shape[1]

        # Extract top-left sub-block (first half of rows and columns)
        top_left = BlocksPoints[i][:rows//2, :cols//2]
        SubBlocksHeights[i].append(top_left.shape[0])
        SubBlocksWidths[i].append(top_left.shape[1])
        top_left = top_left.reshape(-1, len(Elements))
        SubBlocksPoints[i].append(top_left)

        # Extract top-right sub-block (first half of rows and second half of columns)
        top_right = BlocksPoints[i][:rows//2, cols//2:]
        SubBlocksHeights[i].append(top_right.shape[0])
        SubBlocksWidths[i].append(top_right.shape[1])
        top_right = top_right.reshape(-1, len(Elements))
        SubBlocksPoints[i].append(top_right)

        # Extract bottom-left sub-block (second half of rows and first half of columns)
        bottom_left = BlocksPoints[i][rows//2:, :cols//2]
        SubBlocksHeights[i].append(bottom_left.shape[0])
        SubBlocksWidths[i].append(bottom_left.shape[1])
        bottom_left = bottom_left.reshape(-1, len(Elements))
        SubBlocksPoints[i].append(bottom_left)

        # Extract bottom-right sub-block (second half of rows and second half of columns)
        bottom_right = BlocksPoints[i][rows//2:, cols//2:]
        SubBlocksHeights[i].append(bottom_right.shape[0])
        SubBlocksWidths[i].append(bottom_right.shape[1])
        bottom_right = bottom_right.reshape(-1, len(Elements))
        SubBlocksPoints[i].append(bottom_right)
    
    return SubBlocksPoints, SubBlocksHeights, SubBlocksWidths


# --------------------------------------------------
# Visualize the Segmentation Results - No Blocks
# --------------------------------------------------

def PlotClustersNoBlocks(FilePath, Elements, SortedCenters, FirstElementValues):

    # Create a 1x1 grid of subplots
    fig, axs = plt.subplots(1, 1, figsize=(15.3, 7.5))

    # Display the image using 'magma_r' colormap
    im = axs.imshow(FirstElementValues, cmap='magma_r')

    # Set the colorbar for the image
    cbar = plt.colorbar(im, ax=axs)  # Create colorbar for each subplot

    # Extract the first values from SortedCenters and set them as tick positions
    FirstValues = [Values[0] for Values in SortedCenters] 
    cbar.set_ticks(FirstValues)

    # Format the tick labels to display each center in a readable format and set them as tick labels
    FormattedLabels = [f"({', '.join([f'{val:.2f}' for val in Center])})" for Center in SortedCenters]
    cbar.set_ticklabels(FormattedLabels)

    # Set step of 5 on x-axis and y-axis
    axs.xaxis.set_major_locator(MaxNLocator(integer=True, steps=[5]))
    axs.yaxis.set_major_locator(MaxNLocator(integer=True, steps=[5]))

    # Add title based on the number of layers
    if type(Elements) == list and all(type(item) == list for item in Elements):  # Two layers
        StrElements1 = ' ,  '.join(f'"{Element}"' for Element in Elements[0])
        StrElements2 = ' ,  '.join(f'"{Element}"' for Element in Elements[1])
        title = ("Segmentation using K-Means Clustering - Two Layers: " f" {StrElements1}" "  ==> " f" {StrElements2}")
        layer_str = "2L"
    else: # One layer
        StrElements = ' ,  '.join(f'"{Element}"' for Element in Elements)
        title = ("Segmentation using K-Means Clustering - One Layer: " f" {StrElements}")
        layer_str = "1L"
    plt.suptitle(title)

    # Extract the file name from the file path and set it as a subtitle under the title
    file_name = FilePath.split('/')[-1]
    subtitle = ("Dataset Name: " f" {file_name}")
    fig.text(0.5, 0.95, subtitle, ha='center', va='top', fontsize=8)
    
    # Adjust the layout to avoid overlap between plot elements
    plt.tight_layout()

    # Generate custom file name for saving
    k = len(SortedCenters) # Number of clusters
    elements_str = '_'.join([str(e) for e in Elements])  # Concatenate all elements to a string
    timestamp = time.strftime("%Y%m%d-%H%M%S")  # Get the current timestamp
    custom_filename = f"{file_name}_{elements_str}_{layer_str}_k{str(k)}_{timestamp}.png"  # Construct the custom filename
    plt.savefig(custom_filename, dpi=300)  # Save the figure with high resolution

    # Display the figure
    plt.show()


# --------------------------------------------------
# Visualize the Segmentation Results - With Blocks
# --------------------------------------------------

def PlotClustersWithBlocks(FilePath, Elements, SortedCentersBlocks, FirstElementValuesBlocks):

    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(15.3, 7.5))

    # Iterate through each subplot and display the corresponding block of data
    for i, ax in enumerate(axs.flat):

        # Display the image using the 'CMRmap_r' colormap for each subplot
        im = ax.imshow(FirstElementValuesBlocks[i], cmap='CMRmap_r')

        # Set the colorbar for each subplot
        cbar = plt.colorbar(im, ax=ax)

        # Extract the first values from SortedCenters and set them as tick positions
        FirstValues = [Values[0] for Values in SortedCentersBlocks[i]]
        cbar.set_ticks(FirstValues)

        # Format the tick labels to display each center in a readable format and set them as tick labels
        FormattedLabels = [f"({', '.join([f'{val:.2f}' for val in Center])})" for Center in SortedCentersBlocks[i]]
        cbar.set_ticklabels(FormattedLabels)

        # Set step of 5 on x-axis and y-axis
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, steps=[5]))  # Step of 5 on x-axis
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, steps=[5]))  # Step of 5 on y-axis
    
    # Add title based on the number of layers
    if type(Elements) == list and all(type(item) == list for item in Elements):  # Two layers
        StrElements1 = ' ,  '.join(f'"{Element}"' for Element in Elements[0])
        StrElements2 = ' ,  '.join(f'"{Element}"' for Element in Elements[1])
        title = ("Segmentation using K-Means Clustering - Two Layers: " f" {StrElements1}" "  ==> " f" {StrElements2}")
        layer_str = "2L"
    else: # One layer
        StrElements = ' ,  '.join(f'"{Element}"' for Element in Elements)
        title = ("Segmentation using K-Means Clustering - One Layer: " f" {StrElements}")
        layer_str = "1L"
    plt.suptitle(title)

    # Extract the file name from the file path and set it as a subtitle under the title
    file_name = FilePath.split('/')[-1]
    subtitle = ("Dataset Name: " f" {file_name}")
    fig.text(0.5, 0.95, subtitle, ha='center', va='top', fontsize=8)

    # Adjust the layout to avoid overlap between plot elements
    plt.tight_layout()
    plt.tight_layout()
    plt.tight_layout()

    # Generate custom file name for saving
    k = len(SortedCentersBlocks[0])  # Number of clusters
    elements_str = '_'.join([str(e) for e in Elements])  # Concatenate all elements to a string
    timestamp = time.strftime("%Y%m%d-%H%M%S")  # Get the current timestamp
    custom_filename = f"{file_name}_{elements_str}_{layer_str}_k{str(k)}_Blocks_{timestamp}.png"  # Construct the custom filename
    plt.savefig(custom_filename, dpi=300)  # Save the figure with high resolution

    # Display the figure
    plt.show()


# --------------------------------------------------
# Check and Merge the Elements Layers
# --------------------------------------------------

def ProcessLayers(ElementsLayer1, ElementsLayer2):
    
    # Check if Elements_Layer_1 and Elements_Layer_2 are lists
    if type(ElementsLayer1) != list or type(ElementsLayer2) != list:
        print("Please Make Sure that ElementsLayer1 and ElementsLayer2 are lists!")
        return None
    
    # Check if the first layer has at least one element
    if len(ElementsLayer1) == 0:
        print("Please Write at least ONE ELEMENT in the First Layer!")
        return None
    
    # Check if the first layer contains only strings
    for Element in ElementsLayer1:
        if type(Element) != str:
            print("Please make sure that ElementsLayer1 is a list of strings, e.g., ElementsLayer1 = ['Fe', 'S', 'P'].")
            return None
    
    # If the second layer is empty, return only the first layer as a list
    if len(ElementsLayer2) == 0:
        Elements = ElementsLayer1

    # Otherwise, check if the second layer contains only strings and combine both layers into a list of lists
    else:
        for Element in ElementsLayer2:
            # Check if the element is a string
            if type(Element) != str:
                print("Please make sure that ElementsLayer1 is a list of strings, e.g., ElementsLayer2 = ['Fe', 'S', 'P'].")
                return None
        Elements = [ElementsLayer1, ElementsLayer2]
    
    return Elements


# --------------------------------------------------
# Segmantation + Visualization - 1 Layer
# --------------------------------------------------

def Segmentation1Layer(FilePath, ParametersPath, EnergyLevel, Elements, K, DivisionLevel):

    # Get the number of elements and extract their points, height, and width
    m = len(Elements)
    Points, Height, Width = PreLloyd(FilePath, ParametersPath, EnergyLevel, Elements)

    # Check if preprocessing failed (Points will be None if there's an error) and stop the function if so
    if Points is None:
        return
    
    # Check the division level and apply Lloyd's algorithm and visualization accordingly
    # No Blocks - Perform segmentation directly on the entire dataset
    if DivisionLevel == 0:

        # Apply Lloyd's algorithm and get the sorted centers, clustered dataset and first element values
        SortedCenters, ClusteredDataset, FirstElementValues = LloydsAlgorithm(K, Height, Width, m, Points, max_iterations=1000)

        # Plot the segmentation results without any blocks
        PlotClustersNoBlocks(FilePath, Elements, SortedCenters, FirstElementValues)
    
    # 4 Blocks: Divide the dataset into 4 blocks and perform segmentation on each block
    elif DivisionLevel == 1:

        # Initialize empty lists to store sorted centers, clustered datasets, and first element values for each block
        SortedCentersBlocks, ClusteredDatasetBlocks, FirstElementValuesBlocks = [], [], []

        # Divide the dataset into 4 blocks and get their points, heights, and widths
        BlocksPoints, BlocksHeights, BlocksWidths = Division(Points, Height, Width, Elements)

        # Apply Lloyd's algorithm to each of the 4 blocks and store the results
        for i in range(4):
            SortedCenters, ClusteredDataset, FirstElementValues = LloydsAlgorithm(K, BlocksHeights[i], BlocksWidths[i], m, BlocksPoints[i], max_iterations=1000)
            SortedCentersBlocks.append(SortedCenters)
            ClusteredDatasetBlocks.append(ClusteredDataset)
            FirstElementValuesBlocks.append(FirstElementValues)
        
        # Plot the segmentation results with 4 blocks
        PlotClustersWithBlocks(FilePath, Elements, SortedCentersBlocks, FirstElementValuesBlocks)
    
    # 4 Blocks, each Block is divided into 4 SubBlocks (16 SubBlocks in total)
    elif DivisionLevel == 2:

        # Divide the dataset into 4 blocks
        BlocksPoints, BlocksHeights, BlocksWidths = Division(Points, Height, Width, Elements)

        # Further divide each block into 4 sub-blocks
        SubBlocksPoints, SubBlocksHeights, SubBlocksWidths = SubDivision(BlocksPoints, BlocksHeights, BlocksWidths, Elements)

        # iterate through each of the 4 blocks
        for i in range(4):

            # Initialize empty lists to store sorted centers, clustered datasets, and first element values for each sub-block within the current block
            SortedCentersBlocks, ClusteredDatasetBlocks, FirstElementValuesBlocks = [], [], []

            # Iterate through each sub-block within the current block, apply Lloyd's algorithm, and store the results
            for j in range(4):
                SortedCenters, ClusteredDataset, FirstElementValues = LloydsAlgorithm(K, SubBlocksHeights[i][j], SubBlocksWidths[i][j], m, SubBlocksPoints[i][j], max_iterations=1000)
                SortedCentersBlocks.append(SortedCenters)
                ClusteredDatasetBlocks.append(ClusteredDataset)
                FirstElementValuesBlocks.append(FirstElementValues)
            
            # Plot the segmentation results for the current block with 4 sub-blocks
            PlotClustersWithBlocks(FilePath, Elements, SortedCentersBlocks, FirstElementValuesBlocks)


# --------------------------------------------------
# Segmantation + Visualization - 2 Layers
# --------------------------------------------------

def Segmentation2Layers(FilePath, ParametersPath, EnergyLevel, Elements, K1, K2, DivisionLevel):
    
    # Get the number of first layer elements and extract their points, height, and width
    m = len(Elements[0])
    Points, Height, Width = PreLloyd(FilePath, ParametersPath, EnergyLevel, Elements[0])

    # Check if preprocessing failed (Points will be None if there's an error) and stop the function if so
    if Points is None:
        return
    
    # Check the division level and apply Lloyd's algorithm and visualization accordingly
    # No Blocks - Perform segmentation directly on the entire dataset
    if DivisionLevel == 0:

        # Apply Lloyd's algorithm to the first layer elements and get the sorted centers, clustered dataset and first element values
        SortedCenters, ClusteredDataset, FirstElementValues = LloydsAlgorithm(K1, Height, Width, m, Points, max_iterations=1000)

        # Get the number of second layer elements and extract their points, height, and width
        m = len(Elements[1])
        Points, Height, Width = PreLloyd(FilePath, ParametersPath, EnergyLevel, Elements[1])

        # Check if preprocessing failed (Points will be None if there's an error) and stop the function if so
        if Points is None:
            return
        
        # Apply Lloyd's algorithm to the second layer elements and get the sorted centers, clustered dataset and first element values
        SortedCenters, ClusteredDataset, FirstElementValues = LloydLayer2(SortedCenters, ClusteredDataset, K2, Height, Width, m, Points, max_iterations=1000)

        # Plot the segmentation results without any blocks
        PlotClustersNoBlocks(FilePath, Elements, SortedCenters, FirstElementValues)

    # 4 Blocks: Divide the dataset into 4 blocks and perform segmentation on each block
    elif DivisionLevel == 1:

        # Initialize empty lists to store sorted centers, clustered datasets, and first element values for each block
        SortedCentersBlocks, ClusteredDatasetBlocks, FirstElementValuesBlocks = [], [], []

        # Divide the data of the first layer elements into 4 blocks and get their points, heights, and widths
        BlocksPoints, BlocksHeights, BlocksWidths = Division(Points, Height, Width, Elements[0])

        # Apply Lloyd's algorithm to each of the 4 blocks and store the results
        for i in range(4):
            SortedCenters, ClusteredDataset, FirstElementValues = LloydsAlgorithm(K1, BlocksHeights[i], BlocksWidths[i], m, BlocksPoints[i], max_iterations=1000)
            SortedCentersBlocks.append(SortedCenters)
            ClusteredDatasetBlocks.append(ClusteredDataset)
            FirstElementValuesBlocks.append(FirstElementValues)
        
        # Get the number of second layer elements and extract their points, height, and width
        m = len(Elements[1])
        Points, Height, Width = PreLloyd(FilePath, ParametersPath, EnergyLevel, Elements[1])

        # Check if preprocessing failed (Points will be None if there's an error) and stop the function if so
        if Points is None:
            return
        
        # Divide the data of the second layer elements into 4 blocks and get their points, heights, and widths
        BlocksPoints, BlocksHeights, BlocksWidths = Division(Points, Height, Width, Elements[1])

        # Apply Lloyd's algorithm to the second layer elements for each block and store the results
        for i in range(4):
            SortedCenters, ClusteredDataset, FirstElementValues = LloydLayer2(SortedCentersBlocks[i], ClusteredDatasetBlocks[i], K2, BlocksHeights[i], BlocksWidths[i], m, BlocksPoints[i], max_iterations=1000)
            SortedCentersBlocks[i] = SortedCenters
            ClusteredDatasetBlocks[i] = ClusteredDataset
            FirstElementValuesBlocks[i] = FirstElementValues
        
        # Plot the segmentation results with 4 blocks
        PlotClustersWithBlocks(FilePath, Elements, SortedCentersBlocks, FirstElementValuesBlocks)
    
    # 4 Blocks, each Block is divided into 4 SubBlocks (16 SubBlocks in total)
    elif DivisionLevel == 2:

        # Divide the data of the first layer elements into 4 blocks and get their points, heights, and widths
        BlocksPoints, BlocksHeights, BlocksWidths = Division(Points, Height, Width, Elements[0])

        # Further divide each block into 4 sub-blocks and get their points, heights, and widths
        SubBlocksPoints, SubBlocksHeights, SubBlocksWidths = SubDivision(BlocksPoints, BlocksHeights, BlocksWidths, Elements[0])

        # Iterate through each of the 4 blocks of the first layer elements
        for i in range(4):

            # Initialize empty lists to store sorted centers, clustered datasets, and first element values for each sub-block within the current block
            SortedCentersBlocks, ClusteredDatasetBlocks, FirstElementValuesBlocks = [], [], []

            # Iterate through each sub-block within the current block
            for j in range(4):
                # Apply Lloyd's algorithm to the first layer elements for each sub-block and store the results
                SortedCenters, ClusteredDataset, FirstElementValues = LloydsAlgorithm(K1, SubBlocksHeights[i][j], SubBlocksWidths[i][j], m, SubBlocksPoints[i][j], max_iterations=1000)
                SortedCentersBlocks.append(SortedCenters)
                ClusteredDatasetBlocks.append(ClusteredDataset)
                FirstElementValuesBlocks.append(FirstElementValues)
            
            # Get the number of second layer elements and extract their points, height, and width
            m = len(Elements[1])
            Points, Height, Width = PreLloyd(FilePath, ParametersPath, EnergyLevel, Elements[1])

            # Check if preprocessing failed (Points will be None if there's an error) and stop the function if so
            if Points is None:
                return
            
            # Divide the data of the second layer elements into 4 blocks and get their points, heights, and widths
            BlocksPoints, BlocksHeights, BlocksWidths = Division(Points, Height, Width, Elements[1])

            # Further divide each block into 4 sub-blocks and get their points, heights, and widths
            SubBlocksPoints, SubBlocksHeights, SubBlocksWidths = SubDivision(BlocksPoints, BlocksHeights, BlocksWidths, Elements[1])

            # Iterate through each sub-block within the current block of the second layer elements
            for n in range(4):
                # Apply Lloyd's algorithm to the second layer elements for each sub-block and store the results
                SortedCenters, ClusteredDataset, FirstElementValues = LloydLayer2(SortedCentersBlocks[n], ClusteredDatasetBlocks[n], K2, SubBlocksHeights[i][n], SubBlocksWidths[i][n], m, SubBlocksPoints[i][n], max_iterations=1000)
                SortedCentersBlocks[n] = SortedCenters
                ClusteredDatasetBlocks[n] = ClusteredDataset
                FirstElementValuesBlocks[n] = FirstElementValues
            
            # Plot the segmentation results for the current block with 4 sub-blocks
            PlotClustersWithBlocks(FilePath, Elements, SortedCentersBlocks, FirstElementValuesBlocks)



# ==================================================
# 
# Main Function: Segmentation Based on K-means Clustering and Visualization
# 
# ==================================================

def KMeansSegmentation(FilePath, ParametersPath, EnergyLevel, ElementsLayer1, ElementsLayer2, K1, K2, DivisionLevel):

    # Check the elements of the layers and merge them if necessary
    Elements = ProcessLayers(ElementsLayer1, ElementsLayer2)

    # Check if the elements are valid and stop the function if not
    if Elements is None:
        return
    
    # Check if the elements are in a list of lists format (for two layers), if so, call the segmentation function for two layers
    if len(Elements) == 2 and (type(Elements[0]) == list and type(Elements[1]) == list):
        Segmentation2Layers(FilePath, ParametersPath, EnergyLevel, Elements, K1, K2, DivisionLevel)
    
    # Otherwise, call the segmentation function for one layer
    else:
        Segmentation1Layer(FilePath, ParametersPath, EnergyLevel, Elements[0], K1, DivisionLevel)


# ==================================================
#
# IMPLEMENTATION: CALLING THE MAIN FUNCTION
#
# ==================================================


KMeansSegmentation(FilePath, ParametersPath, EnergyLevel, ElementsLayer1, ElementsLayer2, K1, K2, DivisionLevel)


# ==================================================
# ==================================================
#
# END OF THE CODE
#
# ==================================================
# ==================================================