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
import pandas as pd


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
    
    def get_depth(self, iter):
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

    def construct(self, data, threshold=None):
        # This function constructs your tree with a default threshold of None.
        # The depth of the constructed tree is returned.

        print("Here's the first row in the training data:")
        print(data[0])

        # As a starting place, we are statically setting our tree to a root node
        # with two children. This code does not reflect a correct approach as it
        # does not build the tree from the data. (See Decision Trees II Slides.)
        root = InternalNode('Clump Thickness', 0.75)
        child1 = LeafNode(2)  # Decision: Benign
        root.insert_left(child1)

        child2 = LeafNode(4)  # Decision: Malignant
        root.insert_right(child2)

        self.tree = root
        return self.tree.get_depth(0)  # Return the depth of your constructed tree.

    def classify(self, data):
        # This function classifies data with your tree.
        # The predictions for the given data are returned.

        # Use the constructed tree here., e.g. self.tree

        # Return a list of predictions.
        return [2, 2, 4, 2, 3, 4]  # Note: This list should be built dynamically.


# 1. Read in data from file.
print("1. Reading File")
with open("breast-cancer-wisconsin.data") as fp:
    reader = csv.reader(fp, delimiter=",", quotechar='"')
    # Uncomment the following line if the first line in your CSV is column names
    # next(reader, None)  # skip the headers

    # create a list (i.e. array) where each index is a row of the CSV file.
    all_data = [row for row in reader]
print()

# 2. Split the data into training and test sets.
#   Note: This is an example split that simply takes the first 90% of the
#    data read in as training data and uses the remaining 10% as test data. 
print("2. Separating Data")
number_of_rows = len(all_data)  # Get the length of our list.
training_data = all_data
test_data = all_data[:]  #
print()

# 3. Create an instance of the DecisionTreeBuilder class.
print("3. Instantiating DecisionTreeBuilder")
dtb = DecisionTreeBuilder()
print()

# 4. Construct the Tree.
print("4. Constructing the Tree with Training Data")
tree_length = dtb.construct(training_data)
print("Tree Length: " + str(tree_length))
print()

# 5. Classify Test Data using the Tree.
print("5. Classifying Test Data with the Constructed Tree")
predictions = dtb.classify(test_data)
print()

print("-- List of Predictions --")
if (len(predictions) > 0):
    for idx, prediction in enumerate(predictions):
        print("Prediction #" + str(idx) + ': ' + str(prediction))
else:
    print(' : Predictions list is empty.')
