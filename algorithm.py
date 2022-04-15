# Project,phase-1
#   During the first week, you need to complete following data analysis tasks.
#   •Get yourself comfortable with k-means algorithm
#   •Download the data and load it in Python•Impute missing values
#   •Compute data statistics
#   •Plot basic graphs
#   Date: 04/07/2021
#   Author: Mansi

# Project,phase-2
#   The steps in phase 2 include:
#   •Write code for ‘Initial’ step
#   •Write code for ‘Assign’ step
#   •Write code for ‘Recompute’ step
#   Date: 04/15/2021
#   Author: Mansi

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



def initialization(data):
    u2_index, u4_index = np.random.randint(low=0, high=len(data), size=2)
    u2 = data.loc[u2_index,'A2':'A10']
    print("Randomly selected row",u2_index,"for centroid mu_2.")
    print("Initial centroid mu_2:")
    print(u2)
    print("\n")
    u4 = data.loc[u4_index,'A2':'A10']
    print("Randomly selected row",u4_index,"for centroid mu_4.")
    print("Initial centroid mu_4:")
    print(u4)
    return u2,u4 
    
def assign(data, u2, u4):
    subFrame = data.loc[:, 'A2':'A10']
    u2_bucket = []
    u4_bucket = []
    for index in range(0, len(data)):
        row_values = subFrame.iloc[index].values
        u2_distance = np.linalg.norm(row_values - u2)
        u4_distance = np.linalg.norm(row_values - u4)
        u2_bucket.append(index) if u2_distance < u4_distance else u4_bucket.append(index)
    return u2_bucket, u4_bucket

def recompute(data, u2_list, u4_list):
    return data.iloc[u2_list,1:10].mean(), data.iloc[u4_list,1:10].mean()
    

def main():
    col = ["Scn", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "Class"]
    df = pd.read_csv('breast-cancer-wisconsin.data', na_values = '?', names = col)
    df.fillna(method = "ffill")
    for i in col:
        if i in ["Scn", "Class"]:
            continue
        print("Attribute",i,"----------------")
        print("%-25s%-10s" % ("Mean:", np.round(df[i].mean(),1)))
        print("%-25s%-10s" % ("Median:", np.round(df[i].median(),1)))
        print("%-25s%-10s" % ("Variance:", np.round(df[i].var(),1)))
        print("%-25s%-10s" % ("Standard Deviation:", np.round(df[i].std(),1)))
        print("\n")
        
    for j in col:
        if j in ["Scn", "Class"]:
            continue   
        fig = plt.figure() 
        fig.suptitle("Histrogram of attribute "+ j,fontsize = 12)  
        plt.xlabel("Value of attribute")     
        plt.ylabel("Nuber of data points")
        plt.hist(df[j], bins=10, color = "blue", alpha = 0.5, edgecolor="black")
        plt.show()
        
    u2, u4 = initialization(df)
    iteration_index = 0
    prev_u2_list, prev_u4_list = [], []
    while iteration_index < 1500:
        u2_list, u4_list = assign(df, u2, u4)
        if(u2_list == prev_u2_list):
            break
        u2, u4 = recompute(df, u2_list, u4_list)
        prev_u2_list, prev_u4_list = u2_list.copy(), u4_list.copy()
        iteration_index+=1
    print("\nProgram ended after",iteration_index,"iterations.")
    print("Final centroid mu_2:")
    print(u2)
    print("\n")
    print("Final centroid mu_4:")
    print(u4)
    u2_series = pd.Series([2]*len(prev_u2_list), index=prev_u2_list, dtype=pd.Int64Dtype())
    u4_series = pd.Series([4]*len(prev_u4_list), index=prev_u4_list, dtype=pd.Int64Dtype())
    df["Predicted_Class"] = u2_series.add(u4_series, fill_value=0)
    
    print("\nFinal cluster assignment:")
    print("\n")
    print(df.iloc[0:20,[0,10,11]])
main()
