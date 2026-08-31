import numpy as np
import pandas as pd
from decisiontree import DecTree
from multiplereg import MultipleRegression

class RandomForest:
    def __init__(self, tree_count, feature_count, max_depth):
        self.tree_count = tree_count
        self.feature_count = feature_count
        self.max_depth = max_depth
        self.trees = []
        self.rng = np.random.default_rng(67)


    def grow_dem_trees(self, data, answers):
        self.trees = []
        bootstrap_samples = self.rng.integers(low=0, high=data.shape[0], size=(self.tree_count, data.shape[0]))
        outbag = np.zeros((data.shape[0],2))
        # make tree
        for row in bootstrap_samples:
            tree = DecTree(data[row], answers[row], self.max_depth, random_forest=True, random_features_count=self.feature_count)
            self.trees.append(tree)

            #accuracy

            mask = np.ones(data.shape[0], dtype=bool)
            mask[row] = False
            out_bag = np.flatnonzero(mask)

            test_predictions = tree.get_predictions(data[out_bag])
            for idx, val in enumerate(test_predictions):
                if val == 0: 
                    outbag[out_bag[idx], 0] += 1
                else:
                    outbag[out_bag[idx], 1] += 1
        outbag_predictions = np.argmax(outbag, axis=1)

        was_outbagged = np.sum(outbag, axis=1) > 0
        correct = np.sum(outbag_predictions[was_outbagged]==answers[was_outbagged])

        print(f"Out of bag accuracy: {np.sum(correct)/np.sum(was_outbagged)}")

    def predict(self, data):
        return (np.mean([tree.get_predictions(data) for tree in self.trees], axis=0) >= 0.5).astype(int)
        


    # need to code bootstrapping, 



# copied from decisiontree
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
    print(f"Weights/bias: {age_model.weights, age_model.bias}")
    # male = 1, female = 0
    
    shuffled = np.random.default_rng(16).permutation(len(train_data))
    split = int(len(train_data)*0.8)
    train = shuffled[:split]
    validate = shuffled[split:]

    tree_counts = [10, 25, 50, 100, 125, 150, 175, 200, 250, 300, 400, 500, 1000]
    depths = [3, 4, 5, 6, 7, 8, 9]
    for i in depths:
        forest = RandomForest(tree_count=1, feature_count=4, max_depth=i)
        print(i,"Trees",end=" ")
        forest.grow_dem_trees(train_data[train], train_answers[train])

        validation_predictions = forest.predict(train_data[validate])
        print(f"Validation acc: {np.mean(validation_predictions==train_answers[validate])}")




    forest = RandomForest(tree_count=2000, feature_count=4, max_depth=6)
    forest.grow_dem_trees(train_data, train_answers)
    
    ans_df = pd.DataFrame(index=df_test.index, columns=['Survived'], data={'Survived': forest.predict(test_data)})
    #print(ans_df)
    ans_df.to_csv('submissions/titanic_random_forest.csv')
    