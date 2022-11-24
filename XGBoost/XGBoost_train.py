import pandas as pd
import numpy as np
import os

# define the path to the data
train_file = './dataset/NSL_KDD_Train.csv'
test_file = './dataset/NSL_KDD_Test.csv'

# define columns
col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]

# open data files using pandas
train_data = pd.read_csv(train_file, header=None, names=col_names)
test_data = pd.read_csv(test_file, header=None, names=col_names)

# change the labels into "normal", and "attack"
train_data.loc[train_data.label != 'normal', 'label'] = 'attack'
test_data.loc[test_data.label != 'normal', 'label'] = 'attack'

# change the labels into 0 and 1
train_data.loc[train_data.label == 'normal', 'label'] = 0
train_data.loc[train_data.label == 'attack', 'label'] = 1
test_data.loc[test_data.label == 'normal', 'label'] = 0
test_data.loc[test_data.label == 'attack', 'label'] = 1

# for every categorical column, change the data type to category
for col in train_data.columns:
    if train_data[col].dtype == 'object':
        train_data[col] = train_data[col].astype('category')
        test_data[col] = test_data[col].astype('category')

# for every categorical column, change the values into category codes
protocol_categories = train_data.protocol_type.cat.categories.tolist()
protocol_categories += test_data.protocol_type.cat.categories.tolist()
protocol_categories = list(set(protocol_categories))

service_categories = train_data.service.cat.categories.tolist()
service_categories += test_data.service.cat.categories.tolist()
service_categories = list(set(service_categories))

flag_categories = train_data.flag.cat.categories.tolist()
flag_categories += test_data.flag.cat.categories.tolist()
flag_categories = list(set(flag_categories))

train_data['protocol_type'] = train_data['protocol_type'].apply(lambda x: protocol_categories.index(x))
train_data['service'] = train_data['service'].apply(lambda x: service_categories.index(x))
train_data['flag'] = train_data['flag'].apply(lambda x: flag_categories.index(x))

test_data['protocol_type'] = test_data['protocol_type'].apply(lambda x: protocol_categories.index(x))
test_data['service'] = test_data['service'].apply(lambda x: service_categories.index(x))
test_data['flag'] = test_data['flag'].apply(lambda x: flag_categories.index(x))

# change the data type of the label column to int
train_data['label'] = train_data['label'].astype('int')
test_data['label'] = test_data['label'].astype('int')

# change categorical columns into int
train_data['protocol_type'] = train_data['protocol_type'].astype('int')
train_data['service'] = train_data['service'].astype('int')
train_data['flag'] = train_data['flag'].astype('int')

test_data['protocol_type'] = test_data['protocol_type'].astype('int')
test_data['service'] = test_data['service'].astype('int')
test_data['flag'] = test_data['flag'].astype('int')

# print(train_data.head())
# print(test_data.head())

# input and output features
X_train = train_data.drop('label', axis=1)
y_train = train_data['label']
X_test = test_data.drop('label', axis=1)
y_test = test_data['label']

# print the number of normals and attacks in the training data
print('Number of normals in the training data: ', y_train[y_train == 0].count())
print('Number of attacks in the training data: ', y_train[y_train == 1].count())


# this takes the training row and returns a list of the values
# util method
def convert_to_training_format(i):
    i['protocol_type'] = protocol_categories.index(i['protocol_type'])
    i['service'] = service_categories.index(i['service'])
    i['flag'] = flag_categories.index(i['flag'])
    return [i[col_name] for col_name in col_names[:-1]]


# when main code file is run, this code is executed
def main():
    # create an XGBoost model
    import xgboost as xgb
    from sklearn.model_selection import GridSearchCV

    # create a DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # define the parameters for the model
    param = {
        'max_depth': 3,
        'eta': 0.001,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist'
    }

    # number of training iterations
    num_round = 100

    # train the model
    bst = xgb.train(param, dtrain, num_round)

    # make predictions for test data
    preds = bst.predict(dtest)

    # evaluate predictions
    from sklearn.metrics import accuracy_score

    # round predictions
    rounded_preds = [round(value) for value in preds]

    # calculate accuracy
    accuracy = accuracy_score(y_test, rounded_preds)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    # save the model
    import pickle

    # save the model to disk
    filename = 'finalized_model.sav'
    pickle.dump(bst, open(filename, 'wb'))

if __name__ == '__main__':
    main()