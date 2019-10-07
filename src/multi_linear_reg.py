# Importing the libraries
# Libraries for storing the data
import numpy as np
import pandas as pd
# Libraries for visualization
import matplotlib.pyplot as plt
import seaborn as sns
# Libraries for building and evaluating the model
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import math

def read_file(file_name, index_column):
    # wb = pd.read_excel("../mrkt_val.xlsx",index_col="Club")
    wb = pd.read_excel(file_name,index_col=index_column)
    return wb

def train_test_split(wb, features, target):
    """
        wb is a data frame
        features -  the list of the column names
        target - the column to be predicted
    """
    train = wb[:10]
    test =  wb[10:]
    if len(features) is 1:
        X_train = np.array(train[features]).reshape(-1,1)
        X_test = np.array(test[features]).reshape(-1,1)
    else:
        X_train = train[features]
        X_test = test[features]
    
    y_train = np.array(train[target])
    y_test = np.array(test[target])

    return X_train, X_test, y_train, y_test,test.index

def train(X_train, y_train):
    """
        Fits the linear reg model to the given data
        @returns trained Linear regression model
    """
    lineareg = LinearRegression()
    lineareg.fit(X_train, y_train)
    return lineareg

def print_weights(lineareg):
    """
    Prints the model weights
    """
    print("intercept or w0",lineareg.intercept_)
    print("Coefficients",lineareg.coef_)

def test(X_test, lineareg):
    """
        Predicts the y values and returns the predicted values
    """
    pred = lineareg.predict(X_test)
    return pred

def print_results(y_test, pred):
    """
        MAE, RMSE and MSE are calculated here
    """
    mae = metrics.mean_absolute_error(y_test, pred)
    mse = metrics.mean_squared_error(y_test, pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, pred))    
    print("Mean Absolute error = ",mae)
    print("Mean Squared error = ", mse)
    print("Root Mean Squared Error = ",rmse)
    
#     def plot(df,df1):

#     fig, axs = plt.subplots(1,3, figsize=(20,20), sharey=True)
#     plt.title('Actual  Values')
#     axs.scatter(wb2['Marketval'],pred1['Points'])
#     plt.title('Predicted Values')
#     axs.scatter(pred1['Marketval'],pred1['Points'])
#     plt.title('Actual Vs Predicted')
#     axs.scatter(wb2['Points'],pred1['Points'])


def main():

    # A - 1
    wb = read_file('../age_country.xlsx',"Club")
    X_train, X_test, y_train, y_test, test_index = train_test_split(wb, ["Age","F_players"],"Points")
    lineareg = train(X_train, y_train)
    print_weights(lineareg)
    pred = test(X_test, lineareg)
    print_results(y_test, pred)
    # plot(X_test,pred)

    # A - 2
    wb = read_file('../age_country.xlsx',"Club")
    avg_pred = pd.DataFrame({"Predicted_val":np.zeros(20),"count":np.zeros(20)},index=wb.index)    
    for i in range(20):
        wb1 = wb.sample(frac=1)
        print("{} iteration results:\n".format(i))     
        X_train, X_test, y_train, y_test , test_index = train_test_split(wb1, ["Age","F_players"],"Points")              
        lineareg = train(X_train, y_train)
        print_weights(lineareg)
        pred = test(X_test, lineareg)
        avg_pred.loc[test_index , "Predicted_val"] =  np.array(avg_pred.loc[test_index, "Predicted_val"]) + np.array(pred)
        avg_pred.loc[test_index,"count"] = avg_pred.loc[test_index,"count"] + 1    
        print_results(y_test, pred)
    print(avg_pred)
    avg_pred["mean"] = avg_pred["Predicted_val"]/avg_pred["count"]
    print(avg_pred)
    print(wb)


if __name__ == "__main__":
    main()

    



