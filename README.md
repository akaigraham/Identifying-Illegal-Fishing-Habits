# Predicting Fishing Habits to Identify Illegal Fishing Activity

## README Outline
Within this [`README.md`](/README.md) you will find:
1. Introduction
2. Overview of Repository Contents
3. Project Objectives
4. Overview of the Process
5. Findings & Recommendations   
6. Conclusion / Summary

## Introduction
Accurately identify fishing activity using vessel track data alongside various ocean station measurements.  Train, tune, and evaluate a number of machine learning models and techniques, including: Decision Trees, Random Forests, XGBoost, and Neural Networks.

![`Vessels Plotted by Label`](/readme_imgs/map.png)

## Repository Contents
1. [`README.md`](/README.md)
2.

## Project Objectives
Build a classifier to predict whether a vessel is engaged in fishing activity in context of providing policymakers, regulators, and other ocean resource stakeholders a tool to identify illegal fishing activity.  Follow CRISP-DM machine learning process to explore dataset, prepare data for modeling, modeling, and post-model evaluation. Main performance metrics focused on were accuracy and recall as our dataset is not very sensitive to producing false positives.

## Overview of the Process:
Following CRISP-DM, the process follows 5 key steps, including:
1. Business Understanding: Outlines facts and requirements of the project.
2. Data Understanding: focused on unpacking data available and leveraged throughout classification tasks. Section will focus on the distribution of our data, and highlighting relationships between target and predictors.
3. Data Preparation: further preprocessing of our data to prepare for modeling.  Includes separating validation sets, handling missing values, and encoding certain columns
4. Modeling: this section iteratively trains a number of machine learning models, specifically using Decision Trees, Random Forests, XGBoost, and Neural Networks.
5. Evaluation: Final / optimal model is selected and final performance metrics of final model are discussed and evaluated.  Focused primarily on accuracy and recall as performance metrics.

### Business Understanding
According to NOAA, illegal, unreported, and unregulated fishing activities are global problems, threatening ocean ecosystems and sustainable fisheries.  Additionally, these fishing activities threaten economic security and the natural resources that are critical to global food security.  

While ramifications of unregulated and illegal fishing operations are fairly transparent, actual regulation of the world's oceans is far more challenging.  Vast areas of open ocean, lack of clear domain boundaries, and limited resources make it almost impossible to actively monitor protected fishing areas efficiently.  

A machine learning model / classifier that is able to tag vessels based on current fishing status would enable policy makers and regulators to keep a better watch on boats that are actually engaged in fishing activities. Additionally, these policymakers / regulators can cross-reference GPS data of fishing vessels with known regulated areas / restricted zones to ensure illegal activities are not occurring.  

AIS, or Automatic Identification System, is used for tracking marine vessel traffic data. AIS is collected by the US Coast Guard throuh an onboard safety navigation device that transmits and monitors the location and characteristics of large vessels in the US and international waters in real time. In the US, the Coast Guard and commercial vendors collect AIS data.  More information can be found [here](https://marinecadastre.gov/ais/).

Leveraging this AIS data, along with public ocean water conditions from NOAA, I will build a classifier to identify fishing activity.  

### Data Understanding
The two datasets leveraged in this process come from two separate sources:
1. Vessel AIS track data sourced from Global Fishing Watch
2. Various ocean measurements and ocean station data sourced from NOAA and the World Ocean Database

#### AIS Dataset
Global Fishing Watch provides AIS vessel track data for a number of vessel-types, including: drifting longlines, fixed gear, pole and line, purse seines, trawlers, trollers, and others.

After importing and merging all vessel types, the consolidated dataset is fairly large.  The dataset takes up over 2.6GB of storage and contains over 28 million rows.

![AIS Dataset](/imgs/raw_AIS.png)

Once the dataset was loaded, further exploration of the available features was performed, starting with understanding how the target variable and predictor variables are distributed.

![Target](/imgs/target.png)

The majority of values are either 0 (representing not fishing) or 1 (representing fishing).  Values in-between indicate the average score if scored by multiple people.  For this reason, any value between 0 and 1 has some sort of uncertainty from a labeling perspective, and these values will eventually be removed.

![AIS](/imgs/AIS.png)

Reviewing the distribution of predictors gives a better understanding of the data available to us.

![Fishing Rates](/imgs/fishing_rates.png)

Looking at fishing rates by vessel type, we see that just over 60% of all drifting longline timestamps are marked as fishing, which is larger than the other vessels.  Fishing rate is also high among trawlers, trollers, and vessels in the uknown category.

Finally - we can look at how predictors in the AIS dataset relate back to the target variable.  This will help inform how these features can help inform fishing status.

![AIS vs. Price](/imgs/AIS_vs_fishing.png)

#### Ocean Station Measurement Data
Sourcing and importing ocean station measurement data was slightly more challenging.  Original data comes from [NOAA](https://www.ncei.noaa.gov/access/world-ocean-database-select/dbsearch.html).  However, data pulled form the World Ocean Database is encoded per the following [documentation](https://www.ncei.noaa.gov/data/oceans/woa/WOD/DOC/wodreadme.pdf).

As a result, downloaded files are returned in a native `.OSD` file format.  To handle the reading and importing of these native files, I leveraged the `wodpy` library, with more information on `wodpy` available [here](https://github.com/IQuOD/wodpy).  I will also outline the steps I used to pull in the dataset below.

```
# import necessary libraries
from wodpy import wod

# open native .OSD file and test reading of one profile using wodpy
fid = open('data/ocldb1642977297.29281.OSD')
profile = wod.WodProfile(fid) # test reading of .OSD file

# convert the profile to dataframe
ods_profile = profile.df()
```

After running this code and previewing the `ods_profile`, we see the following:

![OSD Profile](/imgs/osd_profile.png)

After successful loading of one OSD profile, we have access to the following information:
* `z`: level depths in meters
* `z_level_qc`: level depth qc flags (0 == all good)
* `z_unc`: depth uncertainty
* `t`: level temperature in Celcius
* `t_level_qc`: level temperature qc flags (0 == all good)
* `t_unc`: temperature uncertainty
* `s`: level salinities
* `s_level_qc`: level salinity qc flags (0 == all good)
* `s_unc`: salinity uncertainty
* `oxygen`: oxygen content (mL / L)
* `phosphate`: phosphate content (uM / L)
* `silicate`: silicate content (uM / L)
* `pH`: pH levels
* `p`: pressure (decibar)

Each profile also has additional information stored in the `.meta` attribute. Calling `ods_profile.meta` gives the following:

![ODS Meta](/imgs/ods_meta.png)

Reviewing both the profile and the meta information, we can see there is relevant information in both places. Understanding this, pulling in the rest of ODS data can be completed using a while loop to extract information from each profile and corresponding meta information.  Code for this step is shown below:

```
# use while loop to pull in all profiles
# WARNING - this will take several minutes to run
ods_df = pd.DataFrame()
n = 1 # used to track loop progress

while ods_profile.is_last_profile_in_file(fid) == False:

    # get df and meta information
    profile_df = ods_profile.df()
    profile_meta = profile_df.meta

    # add columns
    profile_df['lat'] = profile_meta['latitude']
    profile_df['lon'] = profile_meta['longitude']
    profile_df['year'] = profile_meta['year']
    profile_df['month'] = profile_meta['month']
    profile_df['day'] = profile_meta['day']
    profile_df['time'] = profile_meta['time']

    # concat df
    ods_df = pd.concat([profile_df, ods_df], axis=0)

    # update to next ods_profile
    ods_profile = wod.WodProfile(fid)
```

`ods_df` should now look like the following:

![ods_df](/imgs/ods_df.png)

After successfully loading this information, repeat steps taken above to examine the distribution of variables.

![ODS Distribution](/imgs/ODS.png)

#### Merging Datasets
The two datasets are merged using latitude and longitude from both datasets.  Given the amount of data, a ball tree is used to efficiently identify the closest corresponding latitude, longitude pair in the ODS dataset for each AIS latitude, longitude pair.  The ball tree uses Haversine Distance to effectively compute distance between two latitude, longitude pairs.  

I used the following [blog post](https://towardsdatascience.com/using-scikit-learns-binary-trees-to-efficiently-find-latitude-and-longitude-neighbors-909979bd929b) as a reference for efficiently mapping lat, long neighbors using a ball tree.  

```
# import necessary libraries
from math import radians
from sklearn.neighbors import BallTree

# convert coordinates to radians to compute haversine distance
boats_df['lat_radian'] = boats_df['lat'].apply(lambda x: radians(x))
boats_df['lon_radian'] = boats_df['lon'].apply(lambda x: radians(x))
clean_ods_df['lat_radian'] = clean_ods_df['lat'].apply(lambda x: radians(x))
clean_ods_df['lon_radian'] = clean_ods_df['lon'].apply(lambda x: radians(x))

# create ball tree to find min distances and the closest neighbors
ball_tree = BallTree(clean_ods_df[['lat_radian', 'lon_radian']].values, metric='haversine')
dists, idxs = ball_tree.query(boats_df[['lat_radian', 'lon_radian']].values, k=3)
```

3 neighbors were returned for each point - the median value of these three values will be used as the mapped value that is pulled into the AIS dataset.

Now that the indices of the three closest ocean stations is stored in `idxs`, we can loop thru this and compute the median of the three closest stations for each measurement:

```
# go thru each set of neighbors returned, and calc the median value for each new column
# WARNING - will take several minutes to run
depths = []
temps = []
sals = []
oxygens = []
pHs = []
phosphates = []
silicates = []

for index_list in idxs:
    # pull corresponding row to index
    ods_0 = clean_ods_df.iloc[index_list[0]]
    ods_1 = clean_ods_df.iloc[index_list[1]]
    ods_2 = clean_ods_df.iloc[index_list[2]]

    # calc median ods vals
    median_depth = np.median([ods_0['z'], ods_1['z'], ods_2['z']])
    median_temp = np.median([ods_0['t'], ods_1['t'], ods_2['t']])
    median_sal = np.median([ods_0['s'], ods_1['s'], ods_2['s']])
    median_oxygen = np.median([ods_0['oxygen'], ods_1['oxygen'], ods_2['oxygen']])
    median_pH = np.median([ods_0['pH'], ods_1['pH'], ods_2['pH']])
    median_phosphate = np.median([ods_0['phosphate'], ods_1['phosphate'], ods_2['phosphate']])
    median_silicate = np.median([ods_0['silicate'], ods_1['silicate'], ods_2['silicate']])

    # append median vals
    depths.append(median_depth)
    temps.append(median_temp)
    sals.append(median_sal)
    oxygens.append(median_oxygen)
    pHs.append(median_pH)
    phosphates.append(median_phosphate)
    silicates.append(median_silicate)

# create new columns in AIS dataset
boats_df['depth'] = depths
boats_df['temp'] = temps
boats_df['salinity'] = sals
boats_df['oxygen'] = oxygens
boats_df['pH'] = pHs
boats_df['phosphate'] = phosphates
boats_df['silicate'] = silicates
```
Successful running of this should give the following merged dataset, containing both AIS vessel track data, and ocean station measurement data.

![Merged Dataset](/merged_df.png)

As a final data understanding step, comparing ocean station predictors against the target provides gives important information about which features may be most predictive or helpful in the modeling stages.

![AIS vs. Target](/imgs/AIS_vs_fishing.png)


### Data Preparation
With a solid understanding of the data available to us, we can move forward with processing and preparing the consolidated dataset for modeling. As a first step, I handled missing values and any duplicates within the dataset.  

#### Split into Train and Test Sets
Dataset is split into a training and test set for validation purposes later on.

```
# separate out X and y
y = clean_df['is_fishing']
X = clean_df.drop('is_fishing', axis=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    random_state=SEED,
                                                    stratify=y)
```

#### One Hot Encode Categorical Features
```
# instantiate one hot encoder
ohe = OneHotEncoder(sparse=False)

# one hot encode vessel type
vessel_train_ohe = ohe.fit_transform(X_train[['vessel_type']])
vessel_test_ohe = ohe.transform(X_test[['vessel_type']])

# convert back to dataframe
vessel_train_ohe = pd.DataFrame(vessel_train_ohe,
                                columns=ohe.categories_[0],
                                index=X_train.index)

vessel_test_ohe = pd.DataFrame(vessel_test_ohe,
                               columns=ohe.categories_[0],
                               index=X_test.index)
```

#### Scale Features
As a final step in the data preparation phase, all features were scaled to ensure differences in scale do not affect distance-based algorithms.

```
# scale data using standard scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_ohe)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_ohe.columns)
X_test_scaled = scaler.transform(X_test_ohe)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test_ohe.columns)
```

### Modeling
4 different models were evaluated in the modeling section, including decision trees, random forests, XGBoost, and neural networks.

Baseline models were generated for each model for comparison pursposes, with each model being tuned and evaluated using a number of performance criteria.  

Evaluation criteria generated for each model will included the following:
- Accuracy: total number of correct predictions out of total observations.
- Recall: number of true positives out of actual total positives.  Of all boats labeled with `is_fishing=1`, what percentage of them did our model correctly identify as fishing?
- Precision: number of true positives out of predicted positives. Of all boats our model said were fishing, how many times was the boat in question actually fishing?
- F1 Score: harmonic mean of precision and recall. Can't be high unless recall and precision are both high.
- ROC AUC: AUC is an alternative comprehensive metric and ROC graphs allow us to find an optimal precision, recall tradeoff. ROC graphs plot true positive rate vs. false positive rate

In the context of our data, we will want to focus on a model with high recall, and will likely care more about our true positive rate than the false positive rate. It would be better to over-identify potential fishing vessels and mark some as fishing when they are actually not, vs. under-identifying fishing vessels and missing potential illegal fishing activity.  

In the context of neural networks - performance and tuning was performed partially by plotting accuracy for training and validation sets at different epochs.  L1 and L2 Regularization were compared to baseline neural network models.  An example of accuracy plotted by epoch is shown below:

![L2 Regularization](/imgs/L2_reg.png)

### Evaluation
Best overall identified model seen during the modeling phase was a tuned random forest model:
```
# instantiate best overall (random forest)
best_overall = RandomForestClassifier(random_state=SEED,
                                      class_weight='balanced',
                                      n_jobs=-1,
                                      criterion='gini',
                                      max_depth=15,
                                      min_samples_leaf=1,
                                      min_samples_split=2)

best_overall.fit(X_train_final, y_train_final)
```
Printing final results, we see the following performance:
![Best Performance](/imgs/best_performance.png)

## Findings & Recommendations
Summarizing the performance metrics of our final model, we see the following scores:
- Accuracy: 93%, meaning our model assigns the correct label 93% of the time.  A significant improvement over a "dummy" model with accuracy of ~55%.
- Precision: 89%, meaning if our model labels a data point as fishing, there is a 89% chance it is really fishing.
- Recall: 96%, meaning that of all data points actually labeled as fishing, our model was able to correctly identify 96% of them.
- F1 Score: 93%. Represents harmonic mean between recall and precision.  
- ROC AUC: 0.93, represents the area under the ROC curve. A perfect score is 1.

Looking at the confusion matrix for the final model, we can see that there are far more false positives than false negatives, as indicated by the number in the upper right quadrant vs. lower left quadrant. We are more interested in a strong recall, and therefore are not too worried with the level of false positives generated given the end applications.

![`matrix`](/imgs/matrix.png)

Recommendations:
1. Use classifier in conjunction with public AIS data to start tagging ships as fishing or not
2. Cross reference vessels labeled as fishing with illegal fishing zones, regulated areas, etc.
3. Physically search ships arriving at port that have been flagged as potential violators

# Conclusion / Summary
We were able to build a classifier capable of correctly identifying the fishing activity of a vessel 93% of the time. Furthermore, we were fairly strict in addressing overfitting to training data, and results could be further improved going forward if slight overfitting as shown by baseline models is allowed.  
