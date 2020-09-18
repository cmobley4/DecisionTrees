#   author:     Chris Mobley
#   file:       main.py
#   description:
#       this file implements the starter code for Project 1.
#
#   requirements:
#       this file assumes that the 'breast-cancer-wisconsin.data' is
#       located in the same directory
#   
#   resources used for building this starter file
#   - https://bradfieldcs.com/algos/trees/representing-a-tree/
import csv
from math import log2
import pandas as pd
import numpy as np


class InternalNode(object):
    # An Internal Node class that has an associated feature and criteria for splitting. 
    def __init__(self, feature, criteria):  # Constructor
        self.type = type
        self.feature = feature
        self.criteria = criteria
        self.left = None
        self.right = None

    def insert_left(self, child):
        if self.left is None:
            self.left = child
        else:
            child.left = self.left
            self.left = child

    def insert_right(self, child):
        if self.right is None:
            self.right = child
        else:
            child.right = self.right
            self.right = child
    
    def get_depth(self, iter=0):
        # Recursively return the depth of the node.
        l_depth = self.left.get_depth(iter+1)
        r_depth = self.right.get_depth(iter+1)

        # return the highest of the two
        return max([l_depth, r_depth])


class LeafNode(object):
    # A Leaf Node class that has an associated decision.
    def __init__(self, decision):  # Constructor
        self.decision = decision

    def retreiveDecision(self, child):
        return self.decision

    def get_depth(self, iter):
        return iter


class DecisionTreeBuilder:
    # This is a Python class named DecisionTreeBuilder.

    def __init__(self):  # Constructor
        self.tree = None  # Define a ``tree'' instance variable.

    # This function constructs a decision tree with a default threshold of None and a default max depth of None
    # The depth of the constructed tree is returned
    def construct(self, data, threshold=None, max_depth=None, output_feature=None, outputs=(True, False)):
        if self.tree is None:
            self.tree = 0
        else:
            self.tree += 1

        # get output feature name
        # if no output feature is specified assume it is set to the last column name
        if output_feature is None:
            out_feat = data.iloc[:, -1].name
        else:
            out_feat = output_feature

        # Find postive, negative, and total number of samples
        num_samples = data.shape[0]  # total number of samples
        p_samples = data[data[out_feat] == outputs[0]]  # df of all positive samples
        n_samples = data[data[out_feat] == outputs[1]]  # df of all negative samples
        num_p = p_samples.shape[0]  # number of positive samples
        num_n = n_samples.shape[0]  # number of negative samples

        # If all samples are positive, create leaf
        if num_p == num_samples:
            leaf = LeafNode(outputs[0])

            if self.tree == 0:
                self.tree = leaf
                return 0

            self.tree -= 1
            return leaf

        # If all samples are negative, create leaf
        if num_n == num_samples:
            leaf = LeafNode(outputs[1])

            if self.tree == 0:
                self.tree = leaf
                return 0

            self.tree -= 1
            return leaf

        # If tree has reached maximum depth, create leaf
        if max_depth is not None and self.tree == max_depth:
            if num_p > num_n:
                leaf = LeafNode(outputs[0])
            else:
                leaf = LeafNode(outputs[1])

            if self.tree == 0:
                self.tree = leaf
                return 0

            self.tree -= 1
            return leaf

        # Calculate H(S)
        Hs = -(num_p/num_samples)*log2(num_p/num_samples) - (num_n/num_samples)*log2(num_n/num_samples)

        # get list of active features (ie. not including output feature)
        active_features = data.columns.values
        active_features = active_features[active_features != out_feat]

        # Calculate H(S|Feature) and IG for each active feature, keeping track of the feature with the highest IG
        ig_max = 0
        best_feat = None
        split = None
        for feature in active_features:
            # Find splitting value for this feature
            split_value = (p_samples[feature].mean() + n_samples[feature].mean()) / 2.0

            # Find number of positive and negative samples on the left side of split
            p_left = p_samples[p_samples[feature] <= split_value].shape[0]
            n_left = n_samples[n_samples[feature] <= split_value].shape[0]

            # Find number of positive and negative samples on the right side of split
            p_right = p_samples[p_samples[feature] > split_value].shape[0]
            n_right = n_samples[n_samples[feature] > split_value].shape[0]

            # Calculate H(S|Feature)
            H_p_left = 0 if p_left == 0 else -p_left/(p_left + n_left)*log2(p_left/(p_left + n_left))
            H_n_left = 0 if n_left == 0 else -n_left/(p_left + n_left)*log2(n_left/(p_left + n_left))
            H_p_right = 0 if p_right == 0 else -p_right/(p_right + n_right)*log2(p_right/(p_right + n_right))
            H_n_right = 0 if n_right == 0 else -n_right/(p_right + n_right)*log2(n_right/(p_right + n_right))

            Hsf = (p_left + n_left)/num_samples*(H_p_left + H_n_left) +\
                  (p_right + n_right)/num_samples*(H_p_right + H_n_right)

            # get information gain and check if it is the new maximum
            ig = Hs - Hsf
            if ig > ig_max:
                ig_max = ig
                best_feat = feature
                split = split_value

        # if the information gain is below threshold for a new internal node, create leaf
        if threshold is not None and ig_max < threshold:
            if num_p > num_n:
                leaf = LeafNode(outputs[0])
            else:
                leaf = LeafNode(outputs[1])

            if self.tree == 0:
                self.tree = leaf
                return 0

            self.tree -= 1
            return leaf

        node = InternalNode(best_feat, split)

        left_data = data[data[best_feat] <= split]
        left = self.construct(left_data,
                              threshold=threshold, max_depth=max_depth, output_feature=out_feat, outputs=outputs)

        right_data = data[data[best_feat] > split]
        right = self.construct(right_data,
                               threshold=threshold, max_depth=max_depth, output_feature=out_feat, outputs=outputs)

        node.insert_left(left)
        node.insert_right(right)

        if self.tree == 0:
            self.tree = node
            return node.get_depth()

        self.tree -= 1
        return node

    # This function classifies data with your tree.
    # The predictions for the given data are returned.
    def classify(self, data):
        # Iterate through data classifying it based on the tree and storing its predicted value
        predictions = []
        for i, sample in data.iterrows():
            # Work way through nodes until sample reaches a leaf
            node = self.tree
            while isinstance(node, InternalNode):
                if sample[node.feature] > node.criteria:
                    node = node.right
                else:
                    node = node.left

            predictions.append(node.decision)

        # Return a list of predictions.
        return predictions


if __name__ == '__main__':
    # Read training data from file and store in pandas DataFrame
    data = pd.read_csv('E:/Chris Mobley/Documents/Projects/DecisionTrees/Data/breast-cancer-wisconsin-training.data')

    # Build tree based on data
    tree = DecisionTreeBuilder()
    depth = tree.construct(data, output_feature='Class', outputs=(2, 4))
    print('Max Depth:', depth)

    # Test to make sure tree was built correctly
    predictions = tree.classify(data)
    true_values = list(data['Class'].values)
    print(predictions == true_values)
    # d = [['1st', 'Male', 'Child', 'True']] * 5 +\
    #     [['1st', 'Male', 'Adult', 'True']] * 57 + \
    #     [['1st', 'Male', 'Adult', 'False']] * 118 +\
    #     [['1st', 'Female', 'Child', 'True']] * 1 +\
    #     [['1st', 'Female', 'Adult', 'True']] * 140 +\
    #     [['1st', 'Female', 'Adult', 'False']] * 4 +\
    #     [['Lower', 'Male', 'Child', 'True']] * 24 +\
    #     [['Lower', 'Male', 'Child', 'False']] * 35 +\
    #     [['Lower', 'Male', 'Adult', 'True']] * 281 +\
    #     [['Lower', 'Male', 'Adult', 'False']] * 1211 +\
    #     [['Lower', 'Female', 'Child', 'True']] * 27 +\
    #     [['Lower', 'Female', 'Child', 'False']] * 17 +\
    #     [['Lower', 'Female', 'Adult', 'True']] * 176 + \
    #     [['Lower', 'Female', 'Adult', 'False']] * 105
    # data = pd.DataFrame.from_dict({
    #     'Class': [69] * 2201,
    #     'Gender': [69] * 2201,
    #     'Age': [69] * 2201,
    #     'Survived': [69] * 711 + [420] * 1490
    # })
    # tree = DecisionTreeBuilder()
    # print(tree.construct(data, output_feature='Survived', outputs=(69, 420)))

# # 1. Read in data from file.
# print("1. Reading File")
# with open("breast-cancer-wisconsin.data") as fp:
#     reader = csv.reader(fp, delimiter=",", quotechar='"')
#     # Uncomment the following line if the first line in your CSV is column names
#     # next(reader, None)  # skip the headers
#
#     # create a list (i.e. array) where each index is a row of the CSV file.
#     all_data = [row for row in reader]
# print()
#
# # 2. Split the data into training and test sets.
# #   Note: This is an example split that simply takes the first 90% of the
# #    data read in as training data and uses the remaining 10% as test data.
# print("2. Separating Data")
# number_of_rows = len(all_data)  # Get the length of our list.
# training_data = all_data
# test_data = all_data[:]  #
# print()
#
# # 3. Create an instance of the DecisionTreeBuilder class.
# print("3. Instantiating DecisionTreeBuilder")
# dtb = DecisionTreeBuilder()
# print()
#
# # 4. Construct the Tree.
# print("4. Constructing the Tree with Training Data")
# tree_length = dtb.construct(training_data)
# print("Tree Length: " + str(tree_length))
# print()
#
# # 5. Classify Test Data using the Tree.
# print("5. Classifying Test Data with the Constructed Tree")
# predictions = dtb.classify(test_data)
# print()
#
# print("-- List of Predictions --")
# if (len(predictions) > 0):
#     for idx, prediction in enumerate(predictions):
#         print("Prediction #" + str(idx) + ': ' + str(prediction))
# else:
#     print(' : Predictions list is empty.')
