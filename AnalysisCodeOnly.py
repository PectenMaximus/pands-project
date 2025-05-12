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

# Load the dataset
IrisData = pd.read_csv("IrisDataset.csv")

# Drop the 'id' column if it exists
if 'id' in IrisData.columns:
    IrisData = IrisData.drop(columns=['id'])

# Generate summary
summary = IrisData.describe(include='all')

# Define the path where you want to save the text file
save_path = r'C:\AlecProjects\pands-project\Images for Notebook\iris_summary.txt'

# Save the summary to a text file
with open(save_path, "w") as f:
    f.write("Summary of Iris Dataset Variables (excluding 'id'):\n\n")
    f.write(summary.to_string())

# Create a variable for each species that fileters the data based on species name and drop the id column
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

# Define a color palette for the species
species_colors = {'setosa': 'lightblue','versicolor': 'lightgreen','virginica': 'lightcoral'}

# Function to generate histograms for all features for all species with unique colors
def generate_all_histograms(data, save_path):
     # Drop 'id' column 
    if 'id' in data.columns: data = data.drop(columns=['id']) # (COME BACK TO THIS AS IT DOESN'T DROP THE ID COLUMN) (If capital I for Id it creates traceback error)
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

# Define a color palette for the species (must match actual values in the 'Species' column)
species_colors = {'Iris-setosa': 'lightblue','Iris-versicolor': 'lightgreen','Iris-virginica': 'lightcoral'}

# Function to generate a pairplot with species-specific colors
def generate_pairplot(data, save_path):

    # Drop 'id' column if present (COME BACK TO THIS AS ID COLUMN IS STILL SHOWING)
    if 'id' in data.columns:
        data = data.drop(columns=['id'])

    # Generate the pairplot
    pairplot = sns.pairplot(data, hue='Species', palette=species_colors, plot_kws={'edgecolor': 'black'})

    # Move legend outside the plot
    pairplot._legend.set_bbox_to_anchor((1, 1)) # Moves legend to top right
    pairplot._legend.set_loc('upper left')

    # Tighten layout
    plt.tight_layout()

    # Save to file
    plt.savefig(save_path, format='png', bbox_inches='tight')
    print(f"Pairplot saved at: {save_path}")
    plt.show()


# File path to save the image
save_path = r'C:\AlecProjects\pands-project\Images for Notebook\iris_pairplot.png'

# Generate and save the pairplot
generate_pairplot(df, save_path)

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

    # Tighten layout
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

# Load the dataset
df = pd.read_csv("IrisDataset.csv")

# Drop the 'Id' column if it exists
df = df.drop(columns=['Id'], errors='ignore')

# Group by species and calculate the mean of SepalLengthCm and SepalWidthCm
sepal_means = df.groupby('Species')[['SepalLengthCm', 'SepalWidthCm']].mean()

# Find species with the largest and smallest Sepal Length and Width
largest_sepal_length_species = sepal_means['SepalLengthCm'].idxmax()
smallest_sepal_length_species = sepal_means['SepalLengthCm'].idxmin()
largest_sepal_width_species = sepal_means['SepalWidthCm'].idxmax()
smallest_sepal_width_species = sepal_means['SepalWidthCm'].idxmin()

# Print results
print("Heavy Meal:", largest_sepal_length_species)
print("Lite bite:", smallest_sepal_length_species)
print("Heavy Meal:", largest_sepal_width_species)
print("Lite Bite:", smallest_sepal_width_species)

# Load the dataset
df = pd.read_csv("IrisDataset.csv")

# Drop the 'Id' column if it exists
df = df.drop(columns=['Id'], errors='ignore')

# Group by species and calculate the std of SepalLengthCm and SepalWidthCm
petal_std = df.groupby('Species')[['PetalLengthCm', 'PetalWidthCm']].std()

# Find species with the largest and smallest variance in Sepal Length and Width
largest_petal_length_variance = petal_std['PetalLengthCm'].idxmax()
smallest_petal_length_varience = petal_std['PetalLengthCm'].idxmin()
largest_petal_width_varience = petal_std['PetalWidthCm'].idxmax()
smallest_petal_width_varience = petal_std['PetalWidthCm'].idxmin()

# Print results
print("Anybody's guess what you'll get:", largest_petal_length_variance)
print("Safe Bet:", smallest_petal_length_varience)
print("Anybody's guess what you'll get:", largest_petal_width_varience)
print("Safe Bet:", smallest_petal_width_varience)

# Load the dataset
df = pd.read_csv("IrisDataset.csv")
df = df.drop(columns=['Id'], errors='ignore')  # Drop 'Id' column if present

# Ask the user for input
sepal_width = float(input("Enter Sepal Width (cm): "))
sepal_length = float(input("Enter Sepal Length (cm): "))
petal_width = float(input("Enter Petal Width (cm): "))
petal_length = float(input("Enter Petal Length (cm): "))

# Create a DataFrame for the user's input
user_input = pd.Series({'SepalLengthCm': sepal_length,'SepalWidthCm': sepal_width,'PetalLengthCm': petal_length,'PetalWidthCm': petal_width})

# Compute mean values per species
species_means = df.groupby('Species')[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].mean()

# Calculate standard deviation of difference between user's input and each species mean
std_devs = species_means.apply(lambda row: np.std(user_input.values - row.values), axis=1)

# Find the smallest std deviation to the closest matching species
min_std = std_devs.min()

# Print outcome message
if min_std > 1:
    print("That Flower is never going to happen pal")
else:
    print("That Flower could happen")

# Append user input to original dataset for visualization
user_row = user_input.to_dict()
user_row['Species'] = 'User Input'
df = pd.concat([df, pd.DataFrame([user_row])], ignore_index=True)

# Plot using pairplot
palette = {'Iris-setosa': 'lightblue','Iris-versicolor': 'lightgreen','Iris-virginica': 'lightcoral','Dream Flower': 'black'}

sns.pairplot(df, hue='Species', palette=palette, plot_kws={'edgecolor': 'black'}, markers=["o", "o", "o", "X"])
plt.suptitle("Is my Dream Flower Possible?",y = 1) # adjust value for y to move title up
plt.show()

