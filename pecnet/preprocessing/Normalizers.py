import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class Normalizer:
    def __init__(self, normalization_type='standard'):

        if normalization_type == 'standard':
            self.scaler = StandardScaler()
        elif normalization_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("Normalization type must be chosen!")

    def fit(self, data):
        self.scaler.fit(data)

    def transform(self, data):
        return self.scaler.transform(data)

    def fit_transform(self, data):
        return self.scaler.fit_transform(data)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
class MeanNormalizer():

    def __init__(self):
        self.mean = None

    def fit(self, data):
        self.mean = np.mean(data)

    def transform(self, data):
        if self.mean is None:
            raise ValueError("Fit Before Transforming")
        return data - self.mean

    def fit_transform(self, data):
        self.mean = np.mean(data)
        return data - self.mean

class WindowNormalizer():

    def __init__(self):
        self.mean = None

    def fit(self, data):
        self.mean = np.mean(data, 1)

    def transform(self, data):
        if self.mean is None:
            raise ValueError("Fit Before Transforming")
        return data - self.mean[:, np.newaxis]

    def fit_transform(self, data):
        self.mean = np.mean(data, 1)
        return data - self.mean[:, np.newaxis]

    def normalize_with_prewindow(self,data,window_length,step_size=0):
        
        if window_length <= 0 or window_length > len(data):
            raise ValueError("Window length must be positive and less than or equal to the length of the data.")
        if step_size < 0 or step_size > len(data):
            raise ValueError("Step size must be non-negative and less than or equal to the length of the data.")


        normalized_data = []
        normalization_values = []  # List to store the mean values used for normalization
        
        for i in range(len(data)):
                
                start_index = max(0, i - window_length-step_size+1)
                
                if window_length+step_size-1 > i:
                    end_index=i-step_size+1
                else:
                    end_index = start_index+window_length

                if start_index >= end_index:
                    window_mean=0
                else:
                    # Calculate the mean values in the window
                    window_mean = np.mean(data[start_index:end_index])
            
                # Store the mean value
                normalization_values.append(window_mean)

                # Normalize the current value and add it to the normalized_data list
                normalized_value = data[i] - window_mean
                normalized_data.append(normalized_value)

        return normalized_data,normalization_values

class Scaler():
    
        def __init__(self):
            self.scale_coeff=None
        
        def fit_scale1D(self, data,scale_factor):
            self.min = np.min(data)
            self.max = np.max(data)
            self.scale_coeff = (self.max - self.min)*scale_factor
            return data/self.scale_coeff
    
        def scale1D(self, data):
            if self.scale_coeff is None:
                raise ValueError("Fit Before Transforming")
            return data/self.scale_coeff
        
        def unscale1D(self, data):
            if self.scale_coeff is None:
                raise ValueError("Scale Before Transforming")
            return data * self.scale_coeff
    


        
if __name__=="__main__":

    # data=np.array([[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],[10,11],[11,12]])
    # print(data.shape)
    wn = WindowNormalizer()
    # wn.fit(data)
    # print(wn.transform(data))
    # print(data.shape)
    
    # scaler = Scaler()
    # print(scaler.scale1D(np.array([1,2,3,4,5,6,7,8,9,10])))

    nd,np= wn.normalize_with_prewindow([1,2,3,4,5,6,7,8,9,10], 4,1)
    print(nd)