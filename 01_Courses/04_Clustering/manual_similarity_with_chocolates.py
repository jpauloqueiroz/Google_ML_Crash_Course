#%% Import dependencies.
import math
from matplotlib import pyplot as plt
import numpy as np
import numpy.linalg as nla
import pandas as pd
import seaborn as sns
import altair as alt
import re
import pdb  # for Python debugger
import sys
from os.path import join


#%% Change pandas options.
# Set the output display to have one digit for decimal places and limit it to
# printing 15 rows.
np.set_printoptions(precision=2)
pd.options.display.float_format = '{:.2f}'.format
pd.options.display.max_rows = 15


#%% Load and clean the data.
choc_data = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/flavors_of_cacao.csv", sep=",", encoding='latin-1')

# We can rename the columns.
choc_data.columns = ['maker', 'specific_origin', 'reference_number', 'review_date', 'cocoa_percent', 'maker_location', 'rating', 'bean_type', 'broad_origin']

#Replace empty/null values with "Blend".
choc_data["bean_type"] = choc_data["bean_type"].fillna("Blend")

# Cast bean_type to string to remove leading 'u'.
choc_data['bean_type'] = choc_data['bean_type'].astype(str)

# Turn cocoa_percent in a number.
choc_data['cocoa_percent'] = choc_data['cocoa_percent'].str.strip('%')
choc_data['cocoa_percent'] = pd.to_numeric(choc_data['cocoa_percent'])

# Correct spelling mistakes, and replace city with country name.
choc_data['maker_location'] = choc_data['maker_location']\
                            .str.replace('Amsterdam', 'Holland')\
                            .str.replace('U.K.', 'England')\
                            .str.replace('Niacragua', 'Nicaragua')\
                            .str.replace('Domincan Republic', 'Dominican Republic')

# Adding this so that Holland and Netherlands map to the same country.
choc_data['maker_location'] = choc_data['maker_location'].str.replace('Holland', 'Netherlands')

def cleanup_spelling_abbrev(text):
    replacements = [
        ['-', ', '], ['/ ', ', '], ['/', ', '], ['\(', ', '], [' and', ', '], [' &', ', '], ['\)', ''],
        ['Dom Rep|DR|Domin Rep|Dominican Rep,|Domincan Republic', 'Dominican Republic'],
        ['Mad,|Mad$', 'Madagascar, '],
        ['PNG', 'Papua New Guinea, '],
        ['Guat,|Guat$', 'Guatemala, '],
        ['Ven,|Ven$|Venez,|Venez$', 'Venezuela, '],
        ['Ecu,|Ecu$|Ecuad,|Ecuad$', 'Ecuador, '],
        ['Nic,|Nic$', 'Nicaragua, '],
        ['Cost Rica', 'Costa Rica'],
        ['Mex,|Mex$', 'Mexico, '],
        ['Jam,|Jam$', 'Jamaica, '],
        ['Haw,|Haw$', 'Hawaii, '],
        ['Gre,|Gre$', 'Grenada, '],
        ['Tri,|Tri$', 'Trinidad, '],
        ['C Am', 'Central America'],
        ['S America', 'South America'],
        [', $', ''], [',  ', ', '], [', ,', ', '], ['\xa0', ' '],[',\s+', ','],
        [' Bali', ',Bali']
    ]
    for i, j in replacements:
        text = re.sub(i, j, text)
    return text

choc_data['specific_origin'] = choc_data['specific_origin'].str.replace('.', '').apply(cleanup_spelling_abbrev)

# Cast specific_origin to string.
choc_data['specific_origin'] = choc_data['specific_origin'].astype(str)

# Replace null-valued fields with the same value as for specific_origin.
choc_data['broad_origin'] = choc_data['broad_origin'].fillna(choc_data['specific_origin'])

# Clean up spelling mistakes and deal with abbreviations.
choc_data['broad_origin'] = choc_data['broad_origin'].str.replace('.', '').apply(cleanup_spelling_abbrev)

# Change 'Trinitario, Criollo' to "Criollo, Trinitario"
# Check with choc_data['bean_type'].unique()
choc_data.loc[choc_data['bean_type'].isin(['Trinitario, Criollo']),'bean_type'] = "Criollo, Trinitario"
# Confirm with choc_data[choc_data['bean_type'].isin(['Trinitario, Criollo'])]

# Fix chocolate maker names
choc_data.loc[choc_data['maker']=='Shattel','maker'] = 'Shattell'
choc_data['maker'] = choc_data['maker'].str.replace(u'Na\xef\xbf\xbdve','Naive')

# Save the original column names
original_cols = choc_data.columns.values

#print(choc_data)


#%% Preprocess Data.

# Check the review_date.
#sns.distplot(choc_data['review_date'])

# Check the distribution.
#sns.distplot(choc_data['rating'])
# It's a Gaussian. Use Z-score to normalize the data.
choc_data["rating_norm"] = (choc_data["rating"] - choc_data["rating"].mean()) / choc_data["rating"].std()

# Check the cocoa_percent.
#sns.distplot(choc_data["cocoa_percent"])
# Close to Gaussian distribution. Normalize the data.
choc_data["cocoa_percent_norm"] = (choc_data["cocoa_percent"] - choc_data["cocoa_percent"].mean()) / choc_data["cocoa_percent"].std()

#print(choc_data)


#%% Run code to add latitude and longitude data

countries_info = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/countries_lat_long.csv", sep=",", encoding='latin-1')

# Join the chocolate review and geographic information tables on maker country name.
choc_data = pd.merge(choc_data, countries_info, left_on = "maker_location", right_on = "name")
choc_data.rename(columns = {"longitude": "maker_long",
                            "latitude": "maker_lat"}, inplace = True)
choc_data.drop(columns =["name", "country"], inplace = True) # Don't need this data

# Join the chocolate review and geographic information tables on origin country.
choc_data = pd.merge(choc_data, countries_info, left_on = "broad_origin", right_on = "name")
choc_data.rename(columns = {"longitude": "origin_long",
                            "latitude": "origin_lat"}, inplace = True)
choc_data.drop(columns = ["name", "country"], inplace = True) # Don't need this data 

#print(choc_data.head())
#sns.distplot(choc_data["maker_lat"])

numQuantiles = 20
colsQuantiles = ["maker_lat", "maker_long", "origin_lat", "origin_long"]

# Since latitude and longitude don't follow a specific distribution, convert the latitude and longitude information into quantiles.
def createQuantiles(dfColumn, numQuantiles):
    return pd.qcut(dfColumn, numQuantiles, labels = False, duplicates = "drop")

for string in colsQuantiles:
    choc_data[string] = createQuantiles(choc_data[string], numQuantiles)

#print(choc_data.head())

# Quantile values range up to 20. Bring quantile values to the same scale as other feature data by scaling them to [0,1].
def minMaxScaler(numArr):
    minx = np.min(numArr)
    maxx = np.max(numArr)
    numArr = (numArr - minx) / (maxx - minx)
    return numArr

for string in colsQuantiles:
    choc_data[string] = minMaxScaler(choc_data[string])

# The features maker and bean_type are categorical features. Convert categorical features into one-hot encoding.
# Duplicate the "maker" feature since it's removed by one-hot enconding function.
choc_data["maker2"] = choc_data["maker"]
choc_data = pd.get_dummies(choc_data, columns = ["maker2"], prefix = ["maker"], dtype = np.float64)
# Same to "bean_type" feature.
choc_data["bean_type2"] = choc_data["bean_type"]
choc_data = pd.get_dummies(choc_data, columns = ["bean_type2"], prefix = ["bean"], dtype = np.float64)

# Split dataframe into two frames: Original data and data for clustering.
choc_data_backup = choc_data.loc[:, original_cols].copy(deep=True)
choc_data.drop(columns=original_cols, inplace=True)

#print(choc_data.head())


#%% Calculate Manual Similarity.
def getSimilarity(obj1, obj2):
    len1 = len(obj1.index)
    len2 = len(obj2.index)
    if not (len1 == len2):
        print("Error: compared objects must have the same number of features")
        sys.exit()
        return 0
    else:
        similarity = obj1 - obj2
        similarity = np.sum((similarity ** 2.0) / 10.0)
        similarity = 1 - math.sqrt(similarity)
        return similarity

# Calculate the similarity between chocolates.
choc1 = 0
chocsToCompare = [1, 4]

print("Similarity between chocolate " + str(choc1) + " and ...")

for ii in range(chocsToCompare[0], chocsToCompare[1] + 1):
    print(str(ii) + ": " + str(getSimilarity(choc_data.loc[choc1], choc_data.loc[ii])))

print("\n\nFeature data for chocolate " + str(choc1))
print(choc_data_backup.loc[choc1:choc1, :])
print("\n\nFeature data for compared chocolates " + str(chocsToCompare))
print(choc_data_backup.loc[chocsToCompare[0]:chocsToCompare[1], :])


#%% Run cell to setup functions.
def dfSimilarity(df, centroids):
  ### dfSimilarity = Calculate similarities for dataframe input
  ### We need to calculate ||a-b||^2 = |a|^2 + |b|^2 - 2*|a|*|b|
  ### Implement this with matrix operations
  ### See the Appendix for further explanation
  numPoints = len(df.index)
  numCentroids = len(centroids.index)
  ## Strictly speaking, we don't need to calculate the norm of points
  # because it adds a constant bias to distances
  # But calculating it so that the similarity doesn't go negative
  # And that we expect similarities in [0,1] which aids debugging
  pointNorms = np.square(nla.norm(df, axis=1))
  pointNorms = np.reshape(pointNorms, [numPoints, 1])
  ## Calculate the norm of centroids
  centroidNorms = np.square(nla.norm(centroids, axis=1))
  centroidNorms = np.reshape(centroidNorms, (1, numCentroids))
  ## Calculate |a|^2 + |b|^2 - 2*|a|*|b|
  similarities = pointNorms + centroidNorms - 2.0 * np.dot(
      df, np.transpose(centroids))
  # Divide by the number of features
  # Which is 10 because the one-hot encoding means the "Maker" and "Bean" are
  # weighted twice
  similarities = similarities / 10.0
  # numerical artifacts lead to negligible but negative values that go to NaN on the root
  similarities = similarities.clip(min=0.0)
  # Square root since it's ||a-b||^2
  similarities = np.sqrt(similarities)
  return similarities


def initCentroids(df, k, feature_cols):
  # Pick 'k' examples are random to serve as initial centroids
  limit = len(df.index)
  centroids_key = np.random.randint(0, limit - 1, k)
  centroids = df.loc[centroids_key, feature_cols].copy(deep=True)
  # the indexes get copied over so reset them
  centroids.reset_index(drop=True, inplace=True)
  return centroids


def pt2centroid(df, centroids, feature_cols):
  ### Calculate similarities between all points and centroids
  ### And assign points to the closest centroid + save that distance
  numCentroids = len(centroids.index)
  numExamples = len(df.index)
  # dfSimilarity = Calculate similarities for dataframe input
  dist = dfSimilarity(df.loc[:, feature_cols], centroids.loc[:, feature_cols])
  df.loc[:, 'centroid'] = np.argmin(dist, axis=1)  # closest centroid
  df.loc[:, 'pt2centroid'] = np.min(dist, axis=1)  # minimum distance
  return df


def recomputeCentroids(df, centroids, feature_cols):
  ### For every centroid, recompute it as an average of the points
  ### assigned to it
  numCentroids = len(centroids.index)
  for cen in range(numCentroids):
    dfSubset = df.loc[df['centroid'] == cen,
                      feature_cols]  # all points for centroid
    if not (dfSubset.empty):  # if there are points assigned to the centroid
      clusterAvg = np.sum(dfSubset) / len(dfSubset.index)
      centroids.loc[cen] = clusterAvg
  return centroids


def kmeans(df, k, feature_cols, verbose):
  flagConvergence = False
  maxIter = 100
  iter = 0  # ensure kmeans doesn't run for ever
  centroids = initCentroids(df, k, feature_cols)
  while not (flagConvergence):
    iter += 1
    #Save old mapping of points to centroids
    oldMapping = df['centroid'].copy(deep=True)
    # Perform k-means
    df = pt2centroid(df, centroids, feature_cols)
    centroids = recomputeCentroids(df, centroids, feature_cols)
    # Check convergence by comparing [oldMapping, newMapping]
    newMapping = df['centroid']
    flagConvergence = all(oldMapping == newMapping)
    if verbose == 1:
      print('Total distance:' + str(np.sum(df['pt2centroid'])))
    if (iter > maxIter):
      print('k-means did not converge! Reached maximum iteration limit of ' + str(maxIter) + '.')
      sys.exit()
      return
  print ('k-means converged for ' + str(k) + ' clusters' + ' after ' + str(iter) + ' iterations!')
  return [df, centroids]

#%% 
k = 35  #@param

feature_cols = choc_data.columns.values  # save original columns
# initialize every point to an impossible value, the k+1 cluster
choc_data['centroid'] = k
# init the point to centroid distance to an impossible value "2" (>1)
choc_data['pt2centroid'] = 2
[choc_data, centroids] = kmeans(choc_data, k, feature_cols, 1)
print("Data for the first few chocolates, with 'centroid' and 'pt2centroid' on"
      ' the extreme right:')
choc_data.head()
#%%
clusterNumber = 7  #@param
choc_data_backup.loc[choc_data['centroid'] == clusterNumber, :]

#%% Run cell to set up functions { display-mode: "form" }.
def clusterCardinality(df):
  k = np.max(df['centroid']) + 1
  k = k.astype(int)
  print ('Number of clusters:' + str(k))
  clCard = np.zeros(k)
  for kk in range(k):
    clCard[kk] = np.sum(df['centroid'] == kk)
  clCard = clCard.astype(int)
  # print "Cluster Cardinality:"+str(clCard)
  plt.figure()
  plt.bar(range(k), clCard)
  plt.title('Cluster Cardinality')
  plt.xlabel('Cluster Number: ' + str(0) + ' to ' + str(k - 1))
  plt.ylabel('Points in Cluster')
  return clCard


def clusterMagnitude(df):
  k = np.max(df['centroid']) + 1
  k = k.astype(int)
  cl = np.zeros(k)
  clMag = np.zeros(k)
  for kk in range(k):
    idx = np.where(df['centroid'] == kk)
    idx = idx[0]
    clMag[kk] = np.sum(df.loc[idx, 'pt2centroid'])
  # print "Cluster Magnitude:",clMag #precision set using np pref
  plt.figure()
  plt.bar(range(k), clMag)
  plt.title('Cluster Magnitude')
  plt.xlabel('Cluster Number: ' + str(0) + ' to ' + str(k - 1))
  plt.ylabel('Total Point-to-Centroid Distance')
  return clMag


def plotCardVsMag(clCard, clMag):
  plt.figure()
  plt.scatter(clCard, clMag)
  plt.xlim(xmin=0)
  plt.ylim(ymin=0)
  plt.title('Magnitude vs Cardinality')
  plt.ylabel('Magnitude')
  plt.xlabel('Cardinality')


def clusterQualityMetrics(df):
  clCard = clusterCardinality(df)
  clMag = clusterMagnitude(df)
  plotCardVsMag(clCard, clMag)

clusterQualityMetrics(choc_data)

#%% Plot loss vs number of clusters
def lossVsClusters(kmin, kmax, kstep, choc_data):
  kmax += 1  # include kmax-th cluster in range
  kRange = range(kmin, kmax, kstep)
  loss = np.zeros(len(kRange))
  lossCtr = 0
  for kk in kRange:
    [choc_data, centroids] = kmeans(choc_data, kk, feature_cols, 0)
    loss[lossCtr] = np.sum(choc_data['pt2centroid'])
    lossCtr += 1
  plt.scatter(kRange, loss)
  plt.title('Loss vs Clusters Used')
  plt.xlabel('Number of clusters')
  plt.ylabel('Total Point-to-Centroid Distance')


kmin = 5  # @param
kmax = 80  # @param
kstep = 2  # @param
lossVsClusters(kmin, kmax, kstep, choc_data)


# %%
# Calculate distances between "A" point and "B" centroids to return
# axb array where a_i,b_j distance is at (i,j) position
A = np.array([[1, 2, 3],\
              [3, 1, 2],\
              [0, 0, 0]])
A = A / np.max(A)
B = np.array([[4, 5, 6],\
              [6, 6, 6]])
B = B / np.max(B)
numPoints = A.shape[0]
numCentroids = B.shape[0]
pointNorms = np.reshape(nla.norm(A, axis=1)**2.0, [numPoints, 1])
centroidNorms = np.reshape(nla.norm(B, axis=1)**2.0, (1, numCentroids))
"""Distance matrix of size 'p' by 'c' where Distance between point 'p' and centroid 'c' is at (p,c)."""
print (pointNorms + centroidNorms - 2.0 * np.dot(A, np.transpose(B)))

# %%
