"example of creating machine learning models using pandas and scikit-learn"
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def get_data():
    "Creating Data Frame based on csv version of dataset. The dataset can be found using this link: https://www.kaggle.com/datasets/rkiattisak/mobile-phone-price?resource=download"
    data=pd.read_csv("data/Mobile phone price.csv",sep=",")
    return data

def clean_data():
    "Cleaning data, getting rid of Model column and changing type of columns to numerical if it is possible"
    data_2=get_data().copy()
    data_2["Price ($)"]=data_2["Price ($)"].str.replace("$","", regex=True)
    data_2=data_2.rename(columns={"Storage ":"Storage","RAM ":"RAM"})
    data_2["Storage"] = data_2["Storage"].str.replace("GB", "", regex=True)
    data_2["RAM"] = data_2["Storage"].str.replace("GB", "", regex=True)
    data_2["Price ($)"] = data_2["Price ($)"].str.replace(",", ".", regex=True)
    data_2["Screen Size (inches)"] = data_2["Screen Size (inches)"].str.replace("(unfolded)", "",regex=False)
    data_2[["First Screen Size (inches)","Second Screen Size (inches)"]] = data_2["Screen Size (inches)"].str.split('+',expand=True)
    data_2["Second Screen Size (inches)"]=data_2["Second Screen Size (inches)"].fillna("0")
    data_2["Camera (MP)"]=data_2["Camera (MP)"].str.replace("MP","",regex=True)
    data_2["Camera (MP)"] = data_2["Camera (MP)"].str.replace("3D", "0", regex=True)
    data_2["Camera (MP)"] = data_2["Camera (MP)"].str.replace("ToF", "0", regex=True)
    data_2[["First Camera (MP)","Second Camera (MP)","Third Camera (MP)","Fourth Camera (MP)"]]=data_2["Camera (MP)"].str.split('+',expand=True)
    data_2[["Second Camera (MP)","Third Camera (MP)","Fourth Camera (MP)"]] = data_2[["Second Camera (MP)","Third Camera (MP)","Fourth Camera (MP)"]].fillna("0")
    data_2[["Price ($)","Storage","RAM","First Screen Size (inches)","Second Screen Size (inches)","First Camera (MP)","Second Camera (MP)","Third Camera (MP)","Fourth Camera (MP)"]]=data_2[["Price ($)","Storage","RAM","First Screen Size (inches)","Second Screen Size (inches)","First Camera (MP)","Second Camera (MP)","Third Camera (MP)","Fourth Camera (MP)"]].astype(float)
    data_2 = data_2.drop(["Model","Screen Size (inches)","Camera (MP)"], axis=1)
    return data_2

def get_features():
    "Seperating features and target value for prediction. Change the label column"
    data_3=clean_data()
    d=pd.get_dummies(data_3['Brand'])
    d=d.astype(int)
    data_3=pd.concat([data_3,d],axis=1)
    target=data_3["Price ($)"]
    features=data_3.drop(["Price ($)","Brand"],axis=1)
    return features,target

def prediction():
    "Predict price of the phone based on the features"
    x, y=get_features()
    x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    model=LinearRegression()
    model=model.fit(x_train,y_train)
    y_pred=model.predict(x_test).round()
    return y_test, y_pred
def evaluate(y_test, y_pred):
    "Calculate R2 Score of the prediction"
    r=r2_score(y_test,y_pred)
    return r
if __name__ == '__main__':
    y_test, y_pred=prediction()
    r2score=evaluate(y_test,y_pred)
    print(f"R2 Score of the Model: {r2score}")