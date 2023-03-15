# California Housing Prices

The California houses Price dataset is a popular dataset that contains information about housing prices in various neighborhoods in California. This dataset contains information such as the location of the house, the median age of the house, the total number of rooms and bedrooms, the population, the median income of households, and the median house value. This dataset is often used by data analysts and machine learning engineers to build predictive models that can accurately predict the median house value of a particular neighborhood.

In this analysis, we will explore the California houses Price dataset and perform various data cleaning and visualization techniques to better understand the relationships between the different variables in the dataset. We will also use different machine learning algorithms to build a predictive model that can accurately predict the median house value of a neighborhood.

### Section 1: Data Check

In the first section of our analysis, we will perform a data check on the California houses Price dataset. This will include checking the head of the dataset to get an idea of what the data looks like, checking the shape of the dataset to see how many rows and columns it contains, checking for null values, and describing the dataset.

### Section 2: Impute Missing Values

In the second section of our analysis, we will impute missing values in the California houses Price dataset using the K-Nearest Neighbors algorithm. This will help us to fill in any missing data that may be important for our predictive model.

### Section 3: Data Visualization

In the third section of our analysis, we will perform various data visualization techniques to better understand the relationships between the different variables in the dataset. This will include plotting histograms, correlation matrices, and scatter plots of all the variables in the dataset. We will also create maps of California to geographically visualize where each data point is located. Lastly, we will create box plots using the string column 'oceanProximity' to understand the relationship between ocean proximity and others variables.

### Section 4: Building a Predictive Model

In the final section of our analysis, we will use different machine learning algorithms such as Linear Regression, Decision Tree, Random Forest, and Gradient Boosting to build a predictive model that can predict the median house value of a neighborhood. We will compare the performance of each algorithm using R2 and MSE parameters and determine which algorithm is the best for this particular dataset.



## 1.Data check

The first section of our analysis involves a data check of the California houses Price dataset. We will begin by looking at the head of the dataset to get an idea of what the data looks like. This will help us to understand the structure of the dataset and what kind of information it contains.

![Screenshot from 2023-03-15 11-44-32](https://user-images.githubusercontent.com/84750731/225380983-81eb4cb9-07ef-45cd-b9f2-c96740bd5dd7.png)

Next, we will check the shape of the dataset to see how many rows and columns it contains. This will give us an idea of the size of the dataset and how much data we have to work with.

![Screenshot from 2023-03-15 11-44-50](https://user-images.githubusercontent.com/84750731/225381139-9498050d-0658-4a1c-aa48-f68706cb6f86.png)

We will then check for null values in the dataset to see if there are any missing data points. This is important because missing data can affect the accuracy of our predictive model. If we find any null values, we will need to decide how to handle them.

![Screenshot from 2023-03-15 11-45-02](https://user-images.githubusercontent.com/84750731/225381135-a64c63e5-0ff8-47c4-aa08-06a41a072345.png)


Finally, we will describe the dataset to get a better understanding of the distribution of the data. This will provide us with information such as the mean, standard deviation, minimum, and maximum values of each column in the dataset.

![Screenshot from 2023-03-15 11-45-20](https://user-images.githubusercontent.com/84750731/225381131-674b9bfd-cbee-434f-8a47-2d56e625b03d.png)


Overall, this data check will help us to get a better understanding of the California houses Price dataset and prepare us for the next steps in our analysis.







## 2. Data Preparation - KNN algorithm

For the second section of our analysis, we used the K-Nearest Neighbors (KNN) algorithm to impute missing values in the California houses Price dataset. We identified that the 'totalBedrooms' column had missing values that needed to be filled in order to conduct further analysis.

After implementing the KNN algorithm, we were able to fill in the missing values in the 'totalBedrooms' column. This allowed us to move forward with our analysis without any missing data points.

![Screenshot from 2023-03-15 11-45-39](https://user-images.githubusercontent.com/84750731/225384023-b5532715-d931-48d3-99e9-7538b38fe3c8.png)


A figure in our analysis shows that after using the KNN model, there were no missing values in the 'totalBedrooms' column. This figure provides visual evidence of the effectiveness of the KNN algorithm in imputing missing values.

Overall, the KNN algorithm was a useful tool for filling in missing values in the dataset. This allowed us to conduct further analysis with complete data, ensuring the accuracy of our results.



## 3. Exploratory Data
For the third part of our analysis, we will create visualizations to explore the California houses Price dataset. The visualizations will help us to understand the distribution of the data and identify any patterns or relationships between the variables.

First, we will plot histograms of each variable in the dataset using matplotlib. This will allow us to see the distribution of the data for each variable and identify any outliers or unusual values.

<img src="https://user-images.githubusercontent.com/84750731/225171519-f944c8c1-ee29-4a9c-8f43-816407c284f9.png" width="700" height="500" />


Next, we will create a correlation matrix using seaborn. This will allow us to see the correlation between each variable in the dataset. We will use this information to identify any strong correlations that may exist between the variables.

<img src="https://user-images.githubusercontent.com/84750731/225171517-d3991aea-749d-46a9-b17b-274da5b35193.png" width="700" height="500" />

After that, we will create an sns.pairplot() of all the variables in the dataset. This will allow us to see the relationship between each pair of variables in the dataset. We will use this information to identify any trends or patterns that may exist between the variables.

<img src="https://user-images.githubusercontent.com/84750731/225171501-d45b0cfa-17e3-4363-b015-66990259546f.png" width="700" height="500" />

We will also plot the latitude and longitude of each data point on a map of California using basemap. This will allow us to see the geographic distribution of the data and identify any patterns or clusters that may exist.


<p float="left">
  <img src="https://user-images.githubusercontent.com/84750731/225171503-178e3d9e-a1e0-46bf-ab77-9931d884aeca.png" width="400" height="400" />
  <img src="https://user-images.githubusercontent.com/84750731/225171505-ac7b4143-bc45-40fe-8616-58681404536c.png" width="400" height="400" />
</p>

<p float="left">
  <img src="https://user-images.githubusercontent.com/84750731/225171506-14a4974a-8501-4bed-984f-7541ffa09dae.png" width="400" height="400" />
  <img src="https://user-images.githubusercontent.com/84750731/225171507-f0c53121-63d4-49bf-a50e-ca6ff06e44a7.png" width="400" height="400" />
</p>

<p float="left">
  <img src="https://user-images.githubusercontent.com/84750731/225171510-0d7e2232-ea15-443a-a025-5518dbe44c32.png" width="400" height="400" />
  <img src="https://user-images.githubusercontent.com/84750731/225171512-588e58cc-b05a-465f-86fd-9817aa3ec630.png" width="400" height="400" />
</p>

<p float="left">
  <img src="https://user-images.githubusercontent.com/84750731/225171514-d62b61c2-6dd5-437b-8bee-2efff1161b1e.png" width="400" height="400" />
</p>

Finally, we will create boxplots of the 'medianHouseValue', 'meadianIncome' and 'Housingmedianage' column using the 'oceanProximity' column as a grouping variable. This will allow us to see the distribution of the median house value for each category of ocean proximity.

<p float="left">
  <img src="https://user-images.githubusercontent.com/84750731/225171490-190fdfbd-3923-45b5-be1d-4834109c17d3.png" width="450" height="350" />
  <img src="https://user-images.githubusercontent.com/84750731/225171491-d020e93c-69da-478a-8ce6-a165b835a903.png" width="450" height="350" />
</p>
<p float="left">
  <img src="https://user-images.githubusercontent.com/84750731/225171500-a5389795-eac4-40f4-b1e5-6522e30a5039.png" width="450" height="350" />
 </p>

Overall, these visualizations will provide us with a deeper understanding of the California houses Price dataset and help us to identify any relationships or patterns that may exist between the variables.



## 4.MODELS TO PREDICT MEDIAN HOUSE VALUE

### Gradient Boosting
<img src="https://user-images.githubusercontent.com/84750731/225176643-e3d3b5e4-f32b-4788-9428-3a0798d3a848.png" width="400" height="300" />

<img src="https://user-images.githubusercontent.com/84750731/225171473-24069b9e-8b33-463f-b400-7c6951176ce9.png" width="400" height="400" />


### Random Forest
<img src="https://user-images.githubusercontent.com/84750731/225176633-6dc0a7c7-b14f-457e-8348-d1ed484864c0.png" width="400" height="300" />

<img src="https://user-images.githubusercontent.com/84750731/225171480-74ab1027-e373-4f29-ab3a-994e01cca44e.png" width="400" height="400" />


### Decision Tree
<img src="https://user-images.githubusercontent.com/84750731/225176646-9f50ef73-97c7-44f2-a2fc-4a7463bdf791.png" width="400" height="300" />

<img src="https://user-images.githubusercontent.com/84750731/225171483-ca0997eb-c012-417f-a92e-dc63f5adb5c9.png" width="400" height="400" />


### Linear Regression
<img src="https://user-images.githubusercontent.com/84750731/225176640-0805ed0c-26f3-4ff1-88bd-cc8da8648db5.png" width="400" height="300" />

<img src="https://user-images.githubusercontent.com/84750731/225171485-35df8a7d-5a22-4a14-8cd7-6b2af965f377.png" width="400" height="400" />

### Conclusion
![Screenshot from 2023-03-15 11-45-54](https://user-images.githubusercontent.com/84750731/225381123-966b4d31-2e65-4af6-b302-51db4eac46ff.png)

