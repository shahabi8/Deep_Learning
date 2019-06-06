from sklearn.preprocessing import MinMaxScaler

class Data_prep():
    def __init__(self, data):
        self.data = data
        self.scaler = MinMaxScaler()

    def Get_normalised_data(self, index):  
        self.index = index  
        self.data[self.index] = self.scaler.fit_transform(self.data[self.index])
        return self.data

    def Get_de_normalised_data(self, index):
        self.index = index 
        self.data[self.index] = self.scaler.inverse_transform(self.data[self.index])

        return self.data
