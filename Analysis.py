# Imports 

import pandas as pd # We are going to use this module for Data loading, Data Summarisation and Data Cleaning
import numpy as np # We are going to use this module for array manipulation 
import os as os # We are going to use this module to interact with the file system environment 
import matplotlib.pyplot as plt # We are going to use this module for Data Visualisation 
import seaborn as sns # We are just going to use this to make our pairplots

# Get current working directory and path to iris dataset
cwd = os.getcwd()

print("Current working directory:", cwd)

# Get path to IrisDataet.csv file
filename = "IrisDataset.csv"  

# Get absolute path to the file
file_path = os.path.abspath(filename)

print("Full path to the file:", file_path)

# Read Iris Dataset and show first 5 rows
IrisData = pd.read_csv ('IrisDataset.csv')
IrisData.head()

# Search for any null values 
null_values = IrisData.isnull().sum()
print(null_values)

# Show the number and names of the unique iris species in the species column 
unique_species_count = IrisData['Species'].nunique(), IrisData['Species'].unique()
print("Number of unique species:", unique_species_count)

# Show the number of irises in each species 
df = pd.read_csv("IrisDataset.csv")
print(df['Species'].value_counts())

# Describe Iris Dataset
DatasetSummary = IrisData.describe()
print (DatasetSummary)

# Filter the dataset for rows where Species is 'Iris-setosa'
setosa_data = IrisData[IrisData['Species'] == 'Iris-setosa']

# Drop the 'Id' column from the filtered dataset as it is not a variable column 
setosa_data = setosa_data.drop('Id', axis=1)

# Describe the filtered dataset
setosa_summary = setosa_data.describe()
print(setosa_summary)

# Filter the dataset for rows where Species is 'Iris-versicolor'
versicolor_data = IrisData[IrisData['Species'] == 'Iris-versicolor']

# Drop the 'Id' column from the filtered dataset as it is not a variable column 
versicolor_data = versicolor_data.drop('Id', axis=1)

# Describe the filtered dataset
versicolor_summary = versicolor_data.describe()
print(versicolor_summary)
# Filter the dataset for rows where Species is 'Iris-virginica'
virginica_data = IrisData[IrisData['Species'] == 'Iris-virginica']

# Drop the 'Id' column from the filtered dataset as it is not a variable column 
virginica_data = virginica_data.drop('Id', axis=1)

# Describe the filtered dataset
virginica_summary = virginica_data.describe()
print(virginica_summary)

# Read Iris Dataset
#IrisData = pd.read_csv('IrisDataset.csv')

# Get the first 5 rows using head()
#iris_head = IrisData.head()

# Define the path where you want to save the text file (COME BACK AND FIX FILE NOT VISIBLE IN FOLDER)
#save_path = r'C:\AlecProjects\pands-project\Images for Notebook\iris_head_output.txt'

# Create a variable for each species that fileters the data based on species name
setosa_data = df[df['Species'] == 'Iris-setosa']
versicolor_data = df[df['Species'] == 'Iris-versicolor']
virginica_data = df[df['Species'] == 'Iris-virginica']

# Function to generate and display an indivdual histogram for a given variable and species
def generate_histogram(data, feature_name, species_name):
    plt.figure(figsize=(8, 6))
    plt.hist(data[feature_name], bins=10, color='Purple', edgecolor='black')
    plt.title(f'{species_name} - {feature_name} Histogram')
    plt.xlabel(feature_name)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# Generate histogram for  Sepal Length for Setosa as an example
generate_histogram(setosa_data, 'SepalLengthCm', 'Setosa')

# # Define a color palette for the species
species_colors = {'setosa': 'lightblue','versicolor': 'lightgreen','virginica': 'lightcoral'}

# Function to generate histograms for all features for all species with unique colors
def generate_all_histograms(data, save_path):
    species_list = ['setosa', 'versicolor', 'virginica']
    species_data = [setosa_data, versicolor_data, virginica_data]
    
    # Create a figure with subplots for each feature
    num_features = len(data.columns) - 1  # Excluding the 'species' column
    num_species = len(species_list)
    
    # Set up the figure size based on number of features and species
    plt.figure(figsize=(15, num_features * 5)) 
    
    # Loop through each feature and create histograms for each species
    for i, feature_name in enumerate(data.columns[:-1]):  # Skip the 'species' column
        for j, (species, species_df) in enumerate(zip(species_list, species_data)):
            plt.subplot(num_features, num_species, i * num_species + j + 1)  # Create grid of subplots
            # Use unique color for each species
            plt.hist(species_df[feature_name], bins=10, color=species_colors[species], edgecolor='black')
            plt.title(f'{species.capitalize()} - {feature_name}')
            plt.xlabel(feature_name)
            plt.ylabel('Frequency')
            plt.grid(True)
    
    # Adjust layout for better spacing
    plt.tight_layout()
    
    # Save the figure as a PNG file
    plt.savefig(save_path, format='png')
    print(f"Histogram saved at: {save_path}")
    
    # Show the plot (optional, you can comment this out if you don't want it to pop up)
    plt.show()

# Specify the file path to save the PNG file
save_path = r'C:\AlecProjects\pands-project\Images for Notebook\iris_histograms.png'

# Generate histograms for all features for all species and save the result as a PNG
generate_all_histograms(df, save_path)

# Function to generate scatter plots of each pair of variables with custom species colours
def generate_scatter_plots(data, species_colors, save_path=None):

    # Add a 'species_color' column to the DataFrame that maps each species to its corresponding color
    data['species_color'] = data['species'].map(species_colors)
    
    # Generate a pairplot using the custom colors
    pairplot = sns.pairplot(data, hue='species', palette=species_colors, plot_kws={'edgecolor': 'black'})
    
    # Move legend to the upper left outside the plot (It was generating in a bad place originally)
    pairplot._legend.set_bbox_to_anchor((1, 1))  # Move it outside top-right
    pairplot._legend.set_loc('upper left')       # Set to upper left of bounding box

    # Adjusts the laout for better spacing
    plt.tight_layout()
    
    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, format='png')
        print(f"Scatter plots saved at: {save_path}")
    
    # Show the plot
    plt.show()

# Save fikle
save_path = r'C:\AlecProjects\pands-project\Images for Notebook\scatter_plots.png'
generate_scatter_plots(df, species_colors, save_path)

# Function to generate a correlation heatmap
def generate_correlation_heatmap(data, save_path=None):

    # Calculate correlation matrix (excluding non-numeric columns like 'species')
    corr_matrix = data.select_dtypes(include='number').corr()

    # Set up the matplotlib figure
    plt.figure(figsize=(8, 6))

    # Generate a heatmap
    heatmap = sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5)

    # Set plot title
    plt.title('Correlation Heatmap')

    # Adjust layout
    plt.tight_layout()

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight')
        print(f"Correlation heatmap saved at: {save_path}")
    
    # Show the plot
    plt.show()

# YSave file
save_path = r'C:\AlecProjects\pands-project\Images for Notebook\heatmap.png'
generate_correlation_heatmap(df, save_path)