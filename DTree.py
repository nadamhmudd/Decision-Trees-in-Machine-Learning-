import pandas as pd
import numpy as np
from numpy.random import RandomState
from pprint import pprint as display


def splitdataset(data): 
	#split data into training and test sets 
	training = data.sample(frac=1, random_state=RandomState())
	test     = data.loc[~data.index.isin(training.index)]

	'''
	# Separating the target variable 	
	X_train = training.values[:, 1:17]
	Y_train = training.values[:, 0]

	X_test = test.values[:, 1:17]
	Y_test = test.values[:, 0]
	'''

	return training,test


def entropy(column):
    #returns the number of times the values in unique array appears in the column
    elements,counts = np.unique(column,return_counts = True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])

    return entropy


def InformationGain(data,attrb,Y=0): #Y=outcome
    #Calculate the entropy of the parent node
    parent_entropy = entropy(data[Y])
    
    ##Calculate the average entropy of the cheldren nodes
    children,counts= np.unique(data[attrb],return_counts=True)
    average_entropy = np.sum([(counts[i]/np.sum(counts))*entropy(data.where(data[attrb]==children[i]).dropna()[Y]) for i in range(len(children))])
    
    #claculate the information gain
    gain = parent_entropy - average_entropy

    return gain


def BuildingTree(originalData,currData,attributes,Y=0):
    
    #is it pure subset?; outcome only one value
    if len(np.unique(currData[Y])) == 1:
        return np.unique(currData[Y])[0]
    
    else: #apply ID3 Algorithm
        #Select the best attribute for splitting
        gain_arr = [InformationGain(currData,attrb) for attrb in attributes]
        idx = np.argmax(gain_arr) #get tne indx of maximum gain
        splitNode = attributes[idx]
        
        #Create the tree
        tree = {splitNode:{}}
        
        #Remove the selected attribute 
        features = [i for i in attributes if i != splitNode]
        
        #create children tree
        for child in np.unique(currData[splitNode]):
            subData = currData.where(currData[splitNode] == child).dropna()
            
            #recursion with update curr data 
            subtree = BuildingTree(originalData,subData,features)
            
            #Add the sub tree
            tree[splitNode][child] = subtree
            
        return(tree)     

       
def main():   
    #Load Data set then split into training and test sets
	data = pd.read_csv('house-votes-84.data.txt',sep= ',', header = None)
	#print(data.shape)
	training, test= splitdataset(data)
	#building training tree
	tree = BuildingTree(training,training,list(training.columns)[1:]) #skip outcome attribute column[0]
	display(tree)



#calling main function 
main() 
