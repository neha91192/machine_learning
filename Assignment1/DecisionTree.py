import math
import sys

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
    max_depth=4
    selected_mse_feature = set()
    regression = False
    threshold_points = {}


    def __init__(self, is_Regression):
        self.regression = is_Regression

    def build_tree(self, node, depth):
        if node is None:
            return
        data_rows = node.feature_list
        if self.regression:
            if len(data_rows) < 10:
                return
            if depth > self.max_depth:
                return
            else:
                result = self.get_best_feature_mse(len(data_rows))
        else:
            stopping = self.stopping_condition(data_rows, self.labels, depth)
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
            result = self.get_best_feature(len(data_rows), e)

        node.feature_index = result[0]
        node.threshold = result[1]
        left_rows = []
        right_rows = []
        left_sum=0
        right_sum=0
        for i in data_rows:
            if self.training_set[i][node.feature_index] < node.threshold:
                left_rows.append(i)
                left_sum = left_sum + self.labels[i]
            else:
                right_rows.append(i)
                right_sum = right_sum + self.labels[i]

        if len(left_rows) > 0:
            node.left = Node()
            node.left.feature_list = left_rows
            if self.regression:
                node.left.decision = left_sum/len(left_rows)
        if len(right_rows) > 0:
            node.right = Node()
            node.right.feature_list = right_rows
            if self.regression:
                node.right.decision = right_sum/len(right_rows)
        self.build_tree(node.left, depth+1)
        self.build_tree(node.right, depth+1)

        return node

    def stopping_condition(self, data_rows, labels, depth):
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
        if p_spam > 0.75 or p_not_spam>0.75 or depth > self.max_depth or len(data_rows) < 200:
            result[0] = True
            if p_spam>p_not_spam:
                result.append(1.0)
            else:
                result.append(0.0)
        return result


    def get_best_feature(self, root_count, e):
        ig_table = {}
        for feature_id, feature_values in self.feature_set.items():
            if feature_id in self.selected_feature_ids:
                continue
            points = self.get_top_points(feature_values)
            for t in points:
                spam1 = 0
                not_spam1 = 0
                spam2 = 0
                not_spam2 = 0
                #identify if this is working properly
                for idx, val in enumerate(feature_values):
                    if val < t:
                        if self.labels[idx] == 1.0:
                            spam1 = spam1 + 1
                        else:
                            not_spam1 = not_spam1 + 1
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
                #current_ig = self.calculate_ig(spam1, not_spam1, spam2, not_spam2, e1, e2, root_count, e)
                current_ig = e1 + e2
                if feature_id in ig_table:
                    max_ig_value = ig_table[feature_id]
                    if max_ig_value[0] > current_ig:
                        max_ig_value[0] = current_ig
                        max_ig_value[1] = t
                else:
                    max_ig_value = []
                    max_ig_value.append(current_ig)
                    max_ig_value.append(t)
                ig_table[feature_id] = max_ig_value
        max_ig = sys.maxsize
        max_feature = 0
        threshold = 0.0
        for i, val in ig_table.items():
            if val[0] < max_ig and i not in self.selected_feature_ids:
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
        for feature_id, feature_values in self.feature_set.items():
            if feature_id in self.selected_mse_feature:
                continue
            for t in self.threshold_points[feature_id]:
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
                if left_count !=0:
                    left_p = left_value/left_count
                if right_count !=0:
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
                    min_mse_value.append(mse)
                    min_mse_value.append(t)
                min_mse[feature_id] = min_mse_value

        result = []
        min_mse_feature = 0
        min_threshold = 0.1
        min_mse_val = sys.maxsize

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
        #print(len(self.labels))
        self.training_set = train_data
        data_rows = []
        for i in range(len(train_data)):
            data_rows.append(i)
        self.root.feature_list = data_rows
        return self.build_tree(self.root,0)


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
                if i in self.threshold_points:
                    points = self.threshold_points.get(i)
                else:
                    points = set()
                points.add(feature)
                self.threshold_points[i] = points

        label = feature_table[len(feature_table)-1]
        feature_table.pop(len(feature_table)-1)

        return feature_table, label

    def get_top_points(self, points_list):
        result = []
        points_set = set(points_list)
        points = list(points_set)
        points.sort()
        t = 1
        # if(len(points)>400 and len(points)<1001):
        #     t = 2
        # elif(len(points) > 1001 and len(points) < 2001):
        #     t = 4
        # elif (len(points) > 2001 and len(points) < 3001):
        #     t = 5
        # elif (len(points) > 2001 and len(points) < 3001):
        #     t = 6
        # else:
        #     t = 8
        i=1
        while i<len(points)-1:
            result.append(points[i])
            i=i+t
        return result

    # def test(self, classifier, test_data):
    #     square = 0
    #     for i in range(len(test_data)):
    #         temp = classifier
    #         while temp is not None:
    #             feature_index = temp.feature_index
    #             threshold = temp.threshold
    #             if test_data[i][feature_index] < threshold:
    #                 if temp.left is None:
    #                     predicted = temp.decision
    #                     break
    #                 else:
    #                     temp = temp.left
    #             else:
    #                 if temp.right is None:
    #                     predicted = temp.decision
    #                     break
    #                 else:
    #                     temp = temp.right
    #         # actual = self.labels[i]
    #         actual = test_data[i][len(test_data[0]) - 1]
    #         square = square + math.pow((predicted - actual), 2)
    #     print(square/len(test_data))
    #     return square/len(test_data)

    def test(self, classifier, test_data):
        match = 0
        for i in range(len(test_data)):
            temp = classifier
            while temp is not None:
                feature_index = temp.feature_index
                threshold = temp.threshold
                if test_data[i][feature_index] < threshold:
                    if temp.left is None:
                        predicted = temp.decision
                        break
                    else:
                        temp = temp.left
                else:
                    if temp.right is None:
                        predicted = temp.decision
                        break
                    else:
                        temp = temp.right
            actual = test_data[i][len(test_data[0])-1]
            if(predicted == None):
                print("Not assigned")
            if actual == predicted:
                match = match + 1
            # square = square + math.pow((predicted - actual), 2)
        # print(square / len(test_data))
        print(match)
        print(len(test_data))
        accuracy = match/len(test_data)
        print(accuracy)
        return accuracy

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
        ig_value = e - p1*e1 - p2*e2
        return ig_value

