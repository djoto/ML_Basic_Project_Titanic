import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
#from sklearn.linear_model import SGDClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def prepare_data(file_name):
    dataframe = pd.read_csv(file_name)

    dataframe = dataframe.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

    #print(dataframe)

    dataframe = pd.get_dummies(data=dataframe, columns=['Sex', 'Embarked'])
    
    # Set 'Survived' as last column
    columns = list(dataframe.columns)
    if ('Survived' in columns):
        columns.remove('Survived')
        columns.append('Survived')
        dataframe = dataframe[columns]

    dataframe['Age'] = dataframe['Age'] / dataframe['Age'].max()
    dataframe['Fare'] = dataframe['Fare'] / dataframe['Fare'].max()

    #print(dataframe)

    # Check which columns have missing values
    #columns = list(dataframe.columns)
    #for c in columns:
        #cnt = 0
        #for i in range(len(dataframe[c])):
            #if(str(dataframe[c][i]) == 'nan'):
                #cnt += 1
                ##print(str(c)+' '+str(i))
        #print("Num of missing values for "+str(c)+' is '+str(cnt))
        
    dataframe['Age'].fillna(dataframe['Age'].mean(), inplace=True)
    dataframe['Fare'].fillna(dataframe['Fare'].mean(), inplace=True)
    
    return dataframe


dataframe = prepare_data('train.csv')

#print(dataframe)

columns = list(dataframe.columns)
X = dataframe[columns[0:len(columns)-1]]
y = dataframe[columns[len(columns)-1]]

#print(X)
#print(y)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

#model = GaussianNB()
#model = LogisticRegression()
#model = SGDClassifier()
#model = DecisionTreeClassifier()
#model = RandomForestClassifier()
#model = KNeighborsClassifier(n_neighbors=11)
model = SVC(gamma='auto')


model.fit(X_train, y_train)
y_pred = model.predict(X_valid)

accuracy = accuracy_score(y_valid, y_pred)
print(accuracy)
print(confusion_matrix(y_valid, y_pred))


X_test = prepare_data('test.csv')
y_pred = model.predict(X_test)

df = pd.DataFrame({'PassengerId': list(pd.read_csv('test.csv')['PassengerId']), 'Survived': list(y_pred)})
df.to_csv('submission.csv', index=False)

# make submission.zip
#compression_opts = dict(method='zip', archive_name='submission.csv')
#df.to_csv('submission.zip', index=False, compression=compression_opts)



