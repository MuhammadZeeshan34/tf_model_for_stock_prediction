import pandas as pd



class Loader:

    def __init__(self, filename = "", output_column_name = "", percent_test = 10 , input_column_names = []):
        self.input_file = filename
        self.output_column_name = output_column_name
        self.percent_test = percent_test
        self.input_column_names = input_column_names
        self.dataframe = pd.DataFrame


    def read_file(self):
        self.dataframe = pd.read_excel(self.input_file,index_col=None)



    def divide_test_train(self):
        input_size = len(self.dataframe.index)
        test_size = int(input_size *  ( self.percent_test / 100 ))
        train_size = input_size - test_size
        x_training =  self.dataframe.drop(self.output_column_name,axis=1).iloc[:train_size].values
        print(x_training.shape)
        x_test = self.dataframe.drop(self.output_column_name,axis=1).iloc[-test_size:].values
        print(x_test.shape)

        y_training = self.dataframe[[self.output_column_name]].iloc[:train_size].values
        print(y_training.shape)
        y_test = self.dataframe[[self.output_column_name]].iloc[-test_size:].values
        print(y_test.shape)

        return x_training, x_test, y_training, y_test





