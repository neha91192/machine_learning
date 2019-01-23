
'''
for each n-1 split points calculate mean square error. Choose the split which gives minimum error. 
'''
import math


class Node:
    threshold = 0.0
    feature_index = 0
    decision = None
    feature_list = []
    def __init__(self):
        self.left = None
        self.right = None

class DecisionTree:
    root = Node()
    feature_set = {}
    labels = []
    training_set=[]
    selected_feature_ids = set()
    depth=0
    max_depth=10
    selected_mse_feature = set()
    regression = False

    def __init__(self, is_Regression):
        self.regression = is_Regression

    def build_tree(self, node):
        data_rows = node.feature_list
        stopping = self.stopping_condition(data_rows, self.labels)
        if len(stopping) == 0:
            return
        if stopping[0]:
            node.decision = stopping[1]
            return node
        spam_count = 0
        non_spam_count = 0
        for i in data_rows:
            if self.labels[i] == 1.0:
                spam_count= spam_count+1
            elif self.labels[i] == 0.0:
                non_spam_count = non_spam_count+1
        e = self.calculate_entropy(spam_count, non_spam_count)

        if(self.regression):
            result = self.get_best_feature_mse(len(data_rows))
        else:
            result = self.get_best_feature(len(data_rows), e)
        node.feature_index =result[0]
        # node.threshold = result[1]
        node.threshold = 0.3
        left_rows = []
        right_rows = []
        for i in data_rows:
            if self.training_set[i][node.feature_index] < node.threshold:
                left_rows.append(i)
            else:
                right_rows.append(i)
        node.left = Node()
        node.left.feature_list = left_rows
        node.right = Node()
        node.right.feature_list = right_rows
        self.build_tree(node.left)
        self.build_tree(node.right)

        return node

    def stopping_condition(self, data_rows, labels):
        result = []
        if len(data_rows) == 0:
            return result
        spam = 0
        not_spam = 0
        result.append(False)
        for i in data_rows:
            if labels[i] == 1.0:
                spam = spam+1
            else:
                not_spam = not_spam+1
        p_spam = spam/(spam+not_spam)
        p_not_spam = not_spam/(spam+not_spam)
        if p_spam > 0.8 or p_not_spam>0.8:
            result[0] = True
            if p_spam>0.8:
                result.append(1.0)
            else:
                result.append(0.0)
        return result


    def get_best_feature(self, root_count, e):
        ig_table = {}

        for feature_id, feature_values in self.feature_set.items():
            if feature_id in self.selected_feature_ids:
                continue
            t = 0.1
            while t < 1.0:
                spam1 = 0
                not_spam1 = 0
                spam2 = 0
                not_spam2 = 0
                for idx, val in enumerate(feature_values):
                    if val < t:
                        if self.labels[idx] == 1.0:
                            spam1 = spam1 + 1
                        else:
                            not_spam1 = not_spam1
                    else:
                        if self.labels[idx] == 1.0:
                            spam2 = spam2 + 1
                        else:
                            not_spam2 = not_spam2 + 1
                e1 = 0.0
                e2 = 0.0
                if spam1 != 0 and not_spam1 != 0:
                    e1 = self.calculate_entropy(spam1, not_spam1)
                if spam2 !=0 and not_spam2 != 0:
                    e2 = self.calculate_entropy(spam2, not_spam2)

                current_ig = self.calculate_ig(spam1, not_spam1, spam2, not_spam2, e1, e2, root_count, e)
                if feature_id in ig_table:
                    max_ig_value = ig_table[feature_id]
                    if max_ig_value[0] < current_ig:
                        max_ig_value[0] = current_ig
                        max_ig_value[1] = t
                else:
                    max_ig_value = []
                    max_ig_value.append(current_ig)
                    max_ig_value.append(t)
                ig_table[feature_id] = max_ig_value
                t = t + 0.1
        max_ig = 0
        max_feature = 0
        threshold = 0.0
        for i, val in ig_table.items():
            if val[0] > max_ig and i not in self.selected_feature_ids:
                max_feature = i
                max_ig = val[0]
                threshold = val[1]
        self.selected_feature_ids.add(max_feature)
        max_feature_list = []
        max_feature_list.append(max_feature)
        max_feature_list.append(threshold)
        return max_feature_list


    '''
    for each feature and each threshold split, find mse for left, mse for right 
    '''
    def get_best_feature_mse(self, total_root):
        min_mse = {}
        for feature_id, feature_values in self.feature_set:
            if feature_id in self.selected_mse_feature:
                continue
            t=0.1
            while t < 1.0:
                left_value = 0
                left_count = 0
                right_value = 0
                right_count = 0
                left_rows = []
                right_rows = []
                for i, feature in enumerate(feature_values):
                    if feature < t:
                        left_value = left_value+self.labels[i]
                        left_count = left_count+1
                        left_rows.append(i)
                    else:
                        right_value = right_value+self.labels[i]
                        right_count = right_count+1
                        right_rows.append(i)

                left_p = left_value/left_count
                right_p = right_value/right_count

                sum = 0
                for val in left_rows:
                    sum = sum + (self.labels[val]-left_p)*(self.labels[val]-left_p)
                for val in right_rows:
                    sum = sum + (self.labels[val] - right_p) * (self.labels[val] - right_p)

                mse = sum/total_root

                if feature_id in min_mse:
                    min_mse_value = min_mse[feature_id]
                    if min_mse_value[0] > mse:
                        min_mse_value[0] = mse
                        min_mse_value[1] = t
                else:
                    min_mse_value = []
                    min_mse_value[0] = mse
                    min_mse_value[1] = t
                t = t + 0.1

        result = []
        min_mse_feature = 0
        min_threshold = 0.1
        min_mse_val = 0

        for key, value in min_mse.items():
            if value[0] < min_mse_val:
                min_mse_val = value[0]
                min_mse_feature = key
                min_threshold = value[1]
        self.selected_mse_feature.add(min_mse_feature)
        result.append(min_mse_feature)
        result.append(min_threshold)

        return result


    def train(self, train_data):
        feature_set, labels = self.generate_feature_label(train_data)
        self.feature_set = feature_set
        self.labels = labels
        self.training_set = train_data
        data_rows = []
        for i in range(len(train_data)):
            data_rows.append(i)
        self.root.feature_list = data_rows
        return self.build_tree(self.root)


    def generate_feature_label(self, data):
        feature_table = {}
        for entry in data:
            for i, feature in enumerate(entry):
                if i in feature_table:
                    values = feature_table.get(i)
                else:
                    values = []
                values.append(feature)
                feature_table[i] = values
        label = feature_table[len(feature_table)-1]
        feature_table.pop(len(feature_table)-1)

        return feature_table, label


    def test(self, classifier, test_data):
        feature_index = classifier.feature_index
        threshold = classifier.threshold
        square = 0
        for i in range(len(test_data)):
            temp = classifier
            while temp is not None and temp.decision is None:
                if test_data[i][feature_index] < threshold:
                    temp = temp.left
                else:
                    temp = temp.right
            predicted = temp.decision
            actual = test_data[i][len(test_data[0])-1]
            square = square + math.pow((predicted - actual), 2)

        return square/len(test_data)

    def calculate_entropy(self, spam, not_spam):
        p1 = spam/(spam+not_spam)
        p2 = not_spam/(spam+not_spam)
        if p1 == 0:
            p1 = 1
        if p2 == 0:
            p2 = 1
        entropy = -(p1*math.log(p1, 2))-(p2*math.log(p2,2))
        return entropy

    def calculate_ig(self, s1, ns1, s2, ns2, e1, e2, root_count, e):
        p1 = (s1+ns1)/root_count
        p2 = (s2+ns2)/root_count
        #ig_value = self.E - p1*e1 - p2*e2
        # ig_value = p1*(self.E-e1)+p2*(self.E-e2)
        ig_value = e - p1*e1 - p2*e2
        return ig_value

