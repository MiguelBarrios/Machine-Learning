import numpy as np
import pandas as pd

datapath = "/Users/miguelbarrios/Documents/Hands_on_ml/myMLProjects/DataSets/"

#Read in data
df = pd.read_csv(datapath + "Housing.csv")

# saving a dataframe to a csv file
df.to_csv('outfile.csv', encoding = 'utf-8')

# convert a df to a dictionary
dictionary = df.to_dict()

# Peek at the df contents, get collum names and datatypes
df.info

# get summary of column statistics
df.describe()

# get first n rows
df.head(10)

# get last n rows
df.tail(10)

# get (number of rows, number of cols)
df.shape


########### manipulating the data ###############


# getting a subset of collums from a dataset df[[collumname, collumname,...]]
## collum order does not matter
row0 = df[['X1 transaction date', 'X2 house age']]

# Selecting a single row
# if the first row is not numberic use "name of row"
df.loc[5]

# Selecting multiple rows EXP selects row 1,4 and 7
df.loc[[1,4,7]]

# Selecting a range of rows, also works with not numeric data
# ["bob": "Zane"]
# .loc includes the last value with slice notation
df.loc[5:12]

# Selecting row and collum
# Exp selecting the 1,2,4 rows and the 2 collums
# can be used in combinations with slices
df.loc[[1,2,4],['X1 transaction date','X2 house age']]


# Subseting all rows and some collumns
df.loc[:,['X2 house age','X1 transaction date']]

# Subsetting all collums and some rows
df.loc[[1,2,5], :]

# if selecting many rows and colls its cleaner to do the following
rows = [1,2,5,7,3]
cols = ['X2 house age','X1 transaction date']
df.loc[rows,cols]


# If we dont want to use the lables, we can use .iloc instead of .loc
# gest first row
df.iloc[1]

# Deleting a column, when deleting by default always puyt axis = 1
#df = df.drop("X2 house age", axis = 1)

# or
#df = df.drop(columns = "X2 house age")

# Deleting multiple columns
#df = df.drop(['X2 house age','X1 transaction date'], axis = 1)


##### note #####
#when the options (...,inplace = True) is used it preformes operations on the data but nothing is returend
#without it preforms the operations and a new copy of the data is returned
#if using inplace you dont have to resave the df



# Renaming Columns {"cur col name" : "new name",...}
df.rename(columns = {'X2 house age': "house age", "X1 transaction date": "transaction date" }, inplace = True)



# make all headers uppdercase
df.rename(columns = str.upper, inplace = True)
df.rename(columns = str.lower, inplace = True)




print(df.head(5))





