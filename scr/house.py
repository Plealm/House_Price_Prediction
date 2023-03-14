# linear regression, decision trees, or random forests to build your model.
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import contextily as ctx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import KNNImputer
import geopandas as gpd
import geoplot as gplt
import geoplot.crs as gcrs


def info_data(df, nrows=5):
    """Function that explains the data.

    Parameters:
    -----------
    df: pandas.DataFrame


    Returns:
    -----------
    - Head
    - Shape
    - Null columns and count
    - Info
    - Description of the dataset
    """
    # Get basic information about the dataframe
    desc = df.describe(include='all')

    # Print the head
    print(f"First {nrows} rows of the data:\n{df.head(nrows)}\n")

    # Print the shape
    print(f"Shape of the data:\n{desc.loc['count'].to_frame().T}\n")

    # Print the number of null values
    print(f"Null columns and count:\n{df.isnull().sum()}\n")

    # Print the information and description
    if 'dtype' in desc.index:
        print(
            f"Information about the data:\n{desc.loc['dtype'].to_frame().T}\n")
    else:
        print("No information about data types available.\n")

    print(f"Description of the data:\n{desc}\n")


def impute_knn(df):
    """
    Impute missing values in a pandas dataframe using the K-Nearest Neighbors algorithm.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe containing NaN values to be imputed.

    Returns:
    -----------
    pandas.DataFrame
        A new dataframe with NaN values imputed.

    Raises:
    -----------
    ValueError
        If the input dataframe contains non-numeric columns.

    """
    # separate bewteen in numeric and no numeric values
    df_numeric = df.select_dtypes(include=np.number)
    df_non_num = df.select_dtypes(exclude=np.number)

    # check if there are any NaNs
    if df_numeric.isna().sum().sum() == 0:
        return df_numeric

    # apply KNNImputer
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(df_numeric)
    df_imputed = pd.DataFrame(X_imputed, columns=df_numeric.columns)

    return pd.concat([df_imputed, df_non_num], axis=1)


def explore_data(df):
    """
    Plots histograms for all columns in a pandas DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe to be explored.

    Plot:
    --------
    Create histograms for all columns in a pandas DataFrame.

    """
    # create histograms for all columns
    df.hist(figsize=(15, 9), bins=60, color="lightblue", ec="black")
    plt.subplots_adjust(hspace=0.5, wspace=0.4)
    plt.savefig("../images/histograms.png")
    plt.show()


def plot_correlation_matrix(df):
    """
    Plot a correlation matrix heatmap for the numeric features of a pandas DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe.

    Plot:
    --------
    Create correlation matrix from the dataframe.

    """
    # Compute the correlation matrix
    corr_mat = df.corr(numeric_only=True).round(2)

    # Create a mask for the upper triangle of the heatmap
    mask = np.triu(np.ones_like(corr_mat, dtype=bool))

    # Create a heatmap plot using seaborn
    plt.figure(figsize=(17, 10))
    sns.heatmap(corr_mat, mask=mask, cmap='YlGnBu',
                annot=True, cbar=False)
    plt.title('Correlation Matrix', fontsize=18)
    plt.savefig("../images/corrmat.png")
    plt.show()


def geo(df, col):
    """
    Plot a map of California with points representing the locations in the input dataframe,
    colored based on the values in the specified column.

    Args:
        df (pandas.DataFrame): The input dataframe containing location data.
        col (str): The column in the dataframe to use for coloring the points.

    Returns:
        None.
    """
    # Sample 5000 rows from the dataframe
    df = df.sample(n=5000, random_state=2023)

    # Read in the shapefile for California congressional districts
    cali = gpd.read_file(gplt.datasets.get_path(
        'california_congressional_districts'))

    # Calculate the area of each congressional district and add it to the shapefile
    cali = cali.assign(area=cali.geometry.area)

    # Create a geodataframe from the input dataframe with point geometries based on longitude and latitude
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.longitude, df.latitude))

    # Set the projection to Albers Equal Area
    proj = gcrs.AlbersEqualArea(
        central_latitude=37.16611, central_longitude=-119.44944)

    # Create the figure and axis objects
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': proj})

    # Sort the geodataframe by the specified column
    tgdf = gdf.sort_values(by=col, ascending=True)

    # Plot the congressional districts
    cali.plot(ax=ax, color='lightgrey', edgecolor='white')

    # Plot the points with a color gradient based on the specified column
    vmin, vmax = df[col].min(), df[col].max()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap('viridis')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    gdf.plot(column=col, cmap=cmap, norm=norm, ax=ax,
             markersize=3, alpha=1.0)

    # Add the colorbar to the plot
    cbar = plt.colorbar(sm, ax=ax, shrink=0.75)

    # Add Title
    plt.title(
        f"Distribution of {str(col).capitalize().replace('_', ' ')}", fontsize=15)

    # Add basemap
    ctx.add_basemap(ax)

    # Save the figure and show the plot
    plt.tight_layout()
    plt.subplots_adjust(wspace=-0.5)
    plt.savefig(f"../images/geo-{col}.png")
    plt.show()


def reg(df, n):
    """
    Create a pair plot of a random subset of the input dataframe.

    Args:
        df (pandas.DataFrame): The input dataframe containing the data to plot.
        n (int): The number of samples to plot.

    Returns:
        None.
    """
    sns.set_theme(style="ticks")
    # Drop longitude and latitude columns from the dataframe
    columns_drop = ["longitude", "latitude"]
    subset = df.drop(columns=columns_drop)

    # Sample n rows from the dataframe
    subset = subset.sample(n=n, random_state=2023)

    # Create a pair plot
    g = sns.PairGrid(subset, diag_sharey=False)
    g.fig.set_size_inches(14, 13)

    # Plot kernel density estimates on the diagonal
    g.map_diag(sns.kdeplot, lw=2)

    # Plot scatter plots on the lower triangle
    g.map_lower(sns.scatterplot, s=15, edgecolor="k", linewidth=1, alpha=0.4)

    # Plot kernel density estimates on the lower triangle
    g.map_lower(sns.kdeplot, cmap='viridis', n_levels=10)

    # Adjust layout and save plot to file
    plt.tight_layout()
    plt.savefig("../images/plot.png")

    # Show plot
    plt.show()


def create_boxplot(df, col):
    """Create a vertical boxplot of the given column by ocean proximity.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the data.
    col : str
        The name of the column to plot.

    Returns
    -------
    None
    """  # Set the style of the plot
    sns.set_style('whitegrid')

    # Create a subplot figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create the boxplot using seaborn
    sns.boxplot(x='ocean_proximity', y=col, data=df, ax=ax, orient='v')

    # Set the title of the plot
    ax.set_title(
        f"{str(col).capitalize().replace('_', ' ')} by Ocean Proximity")

    # Set the x-axis label
    ax.set_xlabel('Ocean Proximity')

    # Set the y-axis label
    ax.set_ylabel(str(col).capitalize().replace('_', ' '))

    # Save the plot to an image file
    plt.savefig(f"../images/box-{col}.png")

    # Display the plot
    plt.show()


def prediction(df):
    """Train and evaluate different regression models on a given dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the data

    Returns
    -------
        pandas.DataFrame: A table showing the evaluation results for each model.

    """
    # Encode the categorical variable 'ocean_proximity'
    label_encoder = LabelEncoder()
    df['ocean_proximity'] = label_encoder.fit_transform(df['ocean_proximity'])

    # Split the data into training and testing sets
    x = df.drop("median_house_value", axis=1)
    y = df['median_house_value']
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=2023)

    # Scale the numerical features
    independent_scaler = StandardScaler()
    x_train = independent_scaler.fit_transform(x_train)
    x_test = independent_scaler.transform(x_test)

    # Define the regression models to use
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree Regression': DecisionTreeRegressor(random_state=42),
        'Random Forest Regression': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting Regression': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    # Define the evaluation metrics to use
    metrics = {
        'MSE': mean_squared_error,
        'R2': r2_score
    }

    # Train and evaluate each model using the specified metrics
    results = {}
    for model_name, model in models.items():
        model_results = {}
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        for metric_name, metric in metrics.items():
            metric_result = metric(y_test, y_pred)
            model_results[metric_name] = metric_result

        # Add the dictionary of model results to the overall results dictionary
        results[model_name] = model_results

        # Plot the first 50 predictions for each model
        test = pd.DataFrame({'Predicted': y_pred, 'Actual': y_test})
        test = test.reset_index()
        test = test.drop(['index'], axis=1)
        plt.plot(test[:50])
        plt.title(model_name)
        plt.legend(['Actual', 'Predicted'])
        plt.savefig(f"../images/{model_name}_comparation.png")
        joint = sns.jointplot(x='Actual', y='Predicted', data=test, kind="reg")
        joint.fig.suptitle(model_name,
                           weight='bold', size=18)
        joint.fig.tight_layout()
        joint.plot_joint(sns.kdeplot, color="r")
        plt.savefig(f"../images/{model_name}_reg.png")
        plt.show()

    # Convert the results to a Pandas dataframe and round the results to 2 decimal places
    results_df = pd.DataFrame(results).round(2)
    print(results_df)

    # Get the index (model name) that corresponds to the minimum MSE value
    best_mse_model = results_df.iloc[0].idxmin()

    # Get the index (model name) that corresponds to the maximum R2 value
    best_r2_model = results_df.iloc[1].idxmax()

    # Print the column names
    print(f"\nBest model with minimum MSE: {best_mse_model}")
    print(f"Best model with maximum R2: {best_r2_model}")

    return results_df


data = pd.read_csv('../data/housing.csv')
info_data(data)
df = impute_knn(data)
info_data(df)
print(df.isnull().sum())
explore_data(df)
plot_correlation_matrix(df)
cols = df.columns.values
cols = np.delete(cols, [0, 1, -1]).tolist()
for i in cols:
    geo(df, i)
reg(df, 5000)
for j in cols:
    create_boxplot(df, j)
result = prediction(df)
