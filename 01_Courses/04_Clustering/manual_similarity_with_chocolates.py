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


