# LHydra Hybrid Music Recommender System

## Overview

This documentation combines the README.md for setting up the project and the detailed workflow for executing the data analysis. It serves as a complete guide for users to understand the project's structure and the step-by-step analysis process.

## Project Setup

### Getting Started

These instructions will help you get the project up and running on your local machine for development and testing purposes.

#### Prerequisites

You need to install the following software:

- Python (version 3.9 or higher)
- Data analysis libraries: pandas, numpy, matplotlib, seaborn, scikit-learn

#### Installation

Follow these steps to set up your development environment:

1. Clone the repository:
   ```bash
   git clone https://github.com/jnopareboateng/data-analysis-project.git
   ```
2. Navigate to the project directory:
   ```bash
   cd data-analysis-project
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Scripts

You can run the scripts using Jupyter notebooks or execute the Python files directly.

#### Using Jupyter Notebook

- Start the Jupyter notebook server:
  ```bash
  jupyter notebook
  ```
- Open the `.ipynb` file and run the cells sequentially.

#### Using Python Script

- Execute the script with:
  ```bash
  python script_name.py
  ```

## Project Structure

- `dataset.csv`: The original dataset file.
- `cleaned_data.csv`: The dataset after preprocessing.
- `data_analysis.py`: The main script for data analysis.
- `requirements.txt`: The dependencies for the project.

## Detailed Data Analysis Workflow

1. **Initial Setup and Data Loading**

   - Import essential libraries.
   - Load the dataset into a pandas DataFrame.

2. **Data Exploration**

   - Perform basic DataFrame operations.
   - Identify and remove missing values.

3. **Data Preprocessing**

   - Identify and handle duplicate records.
   - One-hot encode categorical variables.
   - Standardize numerical features.

4. **Data Visualization**

   - Calculate descriptive statistics.
   - Generate histograms, box plots, and scatter plots.

5. **Feature Engineering**

   - Create new features.
   - Extract temporal features.
   - Compute advanced audio features.

6. **Verification Checks**

   - Use assertions to ensure feature calculation accuracy.

7. **Feature Selection**

   - Compute a correlation matrix.
   - Assess feature importance with a RandomForestRegressor.

8. **Modeling Preparation**

   - Split the dataset.
   - Define a preprocessing pipeline.
   - Train a machine learning model.

9. **Model Evaluation**

   - Evaluate the model's performance (not included in the code).

10. **Additional Operations**
    - Retrieve specific DataFrame columns.

## Contributing

Please read [CONTRIBUTING.md](https://github.com/jnopareboateng/data-analysis-project/CONTRIBUTING.md) for our code of conduct and pull request process.

## Author

- **Your Name** - [jnopareboateng](https://github.com/jnopareboateng)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- GPT 4o
- GitHub Copilot
- Llama 3
- Francis Martinson
