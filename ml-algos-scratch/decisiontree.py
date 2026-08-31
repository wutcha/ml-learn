import numpy as np
import pandas as pd
import random
from multiplereg import MultipleRegression

def gini_impurity(below_yes, below_no, above_yes, above_no):

    total = below_yes+below_no+above_yes+above_no

    gini_below = 1-(below_yes/(below_yes+below_no))**2-(below_no/(below_yes+below_no))**2
    gini_above = 1-(above_yes/(above_yes+above_no))**2-(above_no/(above_yes+above_no))**2

    return gini_below*(below_yes+below_no)/total + gini_above*(above_yes+above_no)/total
    

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, pred=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.pred = pred
    
class DecTree:
    def __init__(self, data, answers, max_depth, random_forest = False, random_features_count = 0):
        self.random_forest = random_forest
        self.random_features_count = random_features_count
        self.max_depth = max_depth
        self.root = self.build(data, answers, 0)

    def build(self, data, answers, depth):
        if len(np.unique(answers))==1 or depth==self.max_depth or len(data)<=5:
            return Node(pred=(1 if np.mean(answers) >= 0.5 else 0))

        best_feature = -1
        best_thres = 0.0
        smallest_gini = 1.0

        features = range(data.shape[1])
        if self.random_forest == True:
            features = random.sample(range(data.shape[1]),self.random_features_count)

        for feature in features:
            order = np.argsort(data[:,feature])
            sorted_values = data[order,feature]
            sorted_ans = answers[order]

            below_yes = 0
            below_no = 0
            above_yes = np.sum(sorted_ans==1)
            above_no = np.sum(sorted_ans==0)

            min_g=1.0
            thres=0.0

            for i in range(len(sorted_values) - 1):
                if sorted_ans[i] == 1:
                    below_yes += 1
                    above_yes -= 1
                else:
                    below_no += 1
                    above_no -= 1

                if sorted_values[i] == sorted_values[i + 1]:
                    continue

                threshold = (sorted_values[i] + sorted_values[i + 1]) / 2
                gini = gini_impurity(below_yes,below_no,above_yes,above_no)
                if gini < min_g:
                    min_g = gini
                    thres = threshold

            if min_g < smallest_gini:
                best_feature = feature
                best_thres = thres
                smallest_gini = min_g


        # left = true, right = false
        if best_feature == -1:
            #print(df['Survived'].mode()[0])
            return Node(pred=(1 if np.mean(answers)>=0.5 else 0))

        left = self.build(data[data[:,best_feature]<=best_thres], answers[data[:,best_feature]<=best_thres], depth+1)
        right = self.build(data[data[:,best_feature]>best_thres], answers[data[:,best_feature]>best_thres], depth+1)

        #print("BEST:", best_feature, type(best_feature))     
        return Node(feature=best_feature, threshold=best_thres, left=left, right=right, pred = None)

    def predict(self, passenger, node=None):

        if node is None: 
            node = self.root

        if node.pred is not None: 
            return node.pred
        
        #print(node.feature)

        return self.predict(passenger, node.left if passenger[node.feature] <= node.threshold else node.right)

    def get_predictions(self, data):
        return np.array([self.predict(row) for row in data])



df = pd.read_csv('datasets/Titanic-Train-Dataset.csv', index_col=0)
df_test = pd.read_csv('datasets/Titanic-Test-Dataset.csv', index_col=0)

def cleanup(df, features):
    dfr = df.copy()
    dfr['Sex'] = df['Sex'].replace({'male': 1.0, 'female': 0.0})

    age = MultipleRegression()
    known = dfr['Age'].notna()

    age.fit(dfr.loc[known, ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare']].to_numpy(dtype=float), dfr.loc[known, 'Age'].to_numpy(dtype=float))
    dfr.loc[~known, 'Age'] = age.predict(dfr.loc[~known, ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare']].to_numpy(dtype=float))

    data = dfr[features].to_numpy(dtype=float)
    answers = dfr['Survived'].to_numpy()

    return data, answers, age

if __name__=='__main__':
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']

    train_data, train_answers, age_model = cleanup(df, features)
    df_test['Sex'] = df_test['Sex'].replace({'male': 1.0, 'female': 0.0})
    df_test = df_test[features]
    unknown = (df_test['Age'].isna())
    df_test.loc[unknown,'Age'] = age_model.predict(df_test.loc[unknown, ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare']].to_numpy(dtype=float))
    test_data = df_test[features].to_numpy()
    # male = 1, female = 0
    
    tree = DecTree(train_data, train_answers, 5)
    ans_df = pd.DataFrame(index=df_test.index, columns=['Survived'], data={'Survived': tree.get_predictions(test_data)})
    print(ans_df)
    ans_df.to_csv('submissions/titanic_decision_tree.csv')

    # #print(df.isna().sum().sum())
    
    test_data = train_data[700:]
    test_answers = train_answers[700:]
    train_data = train_data[:700]
    train_answers = train_answers[:700]

    test_limits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    for depth in test_limits:
        test_results = DecTree(train_data, train_answers, depth).get_predictions(test_data)
        correct = 0
        
        for idx, val in enumerate(test_results):
            correct += (1 if test_answers[idx] == val else 0)
        
        accuracy = correct/len(test_results)
        print(f"Depth: {depth} | Accuracy: {accuracy}")
    
