
'''
Calculate IG for each feature and sort as per the gain.
'''


'''
for each n-1 split points calculate mean square error. Choose the split which gives minimum error. 
'''
import math

class DecisionTree:
    total_spam_count = 0
    total_not_spam_count = 0
    E=0.0

    class Node:
        def __init__(self, value, feature_index):
            self.left = None
            self.threshold = value
            self.right = None
            self.feature_index = feature_index

    def train(self, feature_set, labels):
        for val in labels:
            if val == 1.0:
                self.total_spam_count= self.total_spam_count+1
            else:
                self.total_not_spam_count = self.total_not_spam_count+1

        # Calculate entropy for root
        self.E = self.calculate_entropy(self.total_spam_count, self.total_not_spam_count)

        ig_table = {}

        for feature_id, feature_values in feature_set.items():
            i = 0.1
            while i < 1.0:
                spam1 = 0
                not_spam1 = 0
                spam2 = 0
                not_spam2 = 0
                for idx, val in enumerate(feature_values):
                    if val < i:
                        if labels[idx] == 1.0:
                            spam1 = spam1+1
                        else:
                            not_spam1 = not_spam1
                    else:
                        if labels[idx] == 1.0:
                            spam2 = spam2+1
                        else:
                            not_spam2 = not_spam2+1
                e1 = 0.0
                e2 = 0.0
                if spam1 and not_spam1 != 0.0:
                    e1 = self.calculate_entropy(spam1, not_spam1)
                if spam2 and not_spam2 != 0.0:
                    e2 = self.calculate_entropy(spam2, not_spam2)

                current_ig = self.calculate_ig(spam1, not_spam1, spam2, not_spam2, e1, e2)
                if feature_id in ig_table:
                    max_ig_value = ig_table[feature_id]
                    if max_ig_value[0] < current_ig:
                        max_ig_value[0] = current_ig
                        max_ig_value[1] = i
                else:
                    max_ig_value = []
                    max_ig_value.append(current_ig)
                    max_ig_value.append(i)
                ig_table[feature_id] = max_ig_value
                i = i+0.1

        max_ig = 0
        max_feature = 0
        threshold=0.0
        for i,val in ig_table.items():
            print(val)
            if val[0] > max_ig:
                max_feature = i
                max_ig = val[0]
                threshold = val[1]

        print(max_feature)
        print(threshold)
        print(max_ig)


    def test(self, decision_tree, test_data):
        print("in test")

    def calculate_entropy(self, spam, not_spam):
        p1 = spam/(spam+not_spam)
        p2 = not_spam/(spam+not_spam)
        if p1 == 0:
            p1 = 1
        if p2 == 0:
            p2 = 1
        entropy = -(p1*math.log(p1, 2))-(p2*math.log(p2,2))
        return entropy

    def calculate_ig(self, s1, ns1, s2, ns2, e1, e2):
        p1 = (s1+ns1)/(self.total_spam_count+self.total_not_spam_count)
        p2 = (s2+ns2)/(self.total_spam_count+self.total_not_spam_count)
        ig_value = p1*(self.E-e1)+p2*(self.E-e2)
        return ig_value