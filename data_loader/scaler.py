from sklearn.preprocessing import MinMaxScaler

def scale_data(x_train, x_test, y_train, y_test, feature_range_start = 0, feature_range_end = 1, scaler_type = 'MinMaxScaler' )
    scaler = MinMaxScaler(feature_range=(feature_range_start,feature_range_end))

    x_scaled_training = scaler.fit_transform(x_train)
    x_scaled_test = scaler.fit_transform(x_test)

    # Test data should be scaled with same amount as train data
    y_scaled_training = scaler.transform(y_train)
    y_scaled_test = scaler.transform(y_test)


    return x_scaled_training, x_scaled_test, y_scaled_training, y_scaled_test