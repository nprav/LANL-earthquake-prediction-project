# -*- coding: utf-8 -*-
"""
Created on Sat May  4 10:58:37 2019

@author: prav9
"""

import os
import pandas as pd
import numpy as np
import psutil
from numpy.fft import rfft, rfftfreq

folder = "temp"
filename = "text.txt"
    
def write_test_file(k,*args):
    """
    Function used to record miscellaneous process information in a status file.
    Used to ensure things are working as expected.
    Presents the parent process id, current process id, and the memory use percentage.
    """
    
    with open(os.path.join(folder,filename),'w') as file:
        file.write('Ppid: {}\n'.format(os.getppid()))
        file.write('Pid: {}\n'.format(os.getpid()))
        file.write('k= {}\n'.format(k))
        for string in args:
            if type(string) == str:
                file.write(string)
        file.write('Memory use = {}%\n'.format(psutil.virtual_memory().percent))
    
        
def truncate_data(k):
    """
    Helper function to read in the k'th training chunk, store every 1000th value, 
    and save it in a separate csv file.
    """
    
    data = pd.read_csv(os.path.join("Data","train_chunk{0}.csv".format(k)))
    data.iloc[::1000,:].to_csv(os.path.join("temp","truncated_chunk{0}.csv".format(k)))
    
    write_test_file(k)
        
        
def data_describe(k):
    """
    Helper function to read in the k'th training chunk and store the data description details.
    """
    
    data = pd.read_csv(os.path.join("Data","train_chunk{0}.csv".format(k)))
    data.iloc[:,1:].describe().to_csv(os.path.join("temp","describe_chunk{0}.csv".format(k)))
    
    write_test_file(k)
    
       
def quake_sample(k,quake_num):
    """
    Helper function to search and store neighborhoods of the 16 labquakes in the dataset.
    """
    
    # Read in the chunk of training data
    data = pd.read_csv(os.path.join("Data","train_chunk{0}.csv".format(k)))

    # Further explore the dataset if the range in time_to_failure exceeds 1; ie there is a labquake
    if (data['time_to_failure'].max() - data['time_to_failure'].min()) > 1:
        segment_size = 150000
        
        # Iterate through the dataset in segments of 150000 rows to find the labquake
        for i in range(int(np.ceil(data.shape[0]/segment_size))):
            start = i*segment_size
            stop = min((i+1)*segment_size, data.shape[0])
            
            # IF the labquake is found, save the 150000 row neighborhood to a separate csv file
            if (data.loc[start:stop,'time_to_failure'].max() - data.loc[start:stop,'time_to_failure'].min()) > 1:
                data.loc[start:stop].to_csv(os.path.join("temp","quake{0}.csv".format(quake_num.value)))
                quake_num.value += 1
                
    write_test_file(k,'quake_num = {}\n'.format(quake_num.value))


#------------------- Functions to generate feature sets from the dataset ------------------------------#
        
def gen_features1(x_raw):
    """
    Helper function to generate simple statistical features from a 150,000 row 'acoustic data' segment.
    x_raw must be a pandas series
    """
    
    feature_names = ['mean', 'std', 'max', 'min', 'skew', 'kurtosis','q25','q75']

    features = [x_raw.mean(),
                x_raw.std(),
                x_raw.max(),
                x_raw.min(),
                x_raw.skew(),
                x_raw.kurtosis(),
                x_raw.quantile(0.25),
                x_raw.quantile(0.75),
            ]
    
    return features, feature_names


def gen_features3(x_raw,feat_divisions=[1e4,6e4,1.2e5,1.6e5,2e5,2.3e5,3e5],sample_freq = 4e6):
    """
    Helper function to generate simple statistical features and fft amplitudes
    from a 150,000 row 'acoustic data' segment.
    
    x_raw is the input acoustic_data pandas series
    feat_divisions is the ranges of frequencies in Hz to store fourier coefficients for.
    sample_freq is the sampling frequency of the provided data. The default is 4MHz.
    
    returns the converted features and the names of the names of the features.
    """
    
    feature_names = ['mean', 'std', 'max', 'min', 'skew', 'kurtosis']
    feature_names += ["fft_[{:},{:}]Hz".format(int(i),int(j)) for (i,j) in zip(feat_divisions,feat_divisions[1:])]
    
    features = [x_raw.mean(),
                x_raw.std(),
                x_raw.max(),
                x_raw.min(),
                x_raw.skew(),
                x_raw.kurtosis(),
            ]

    freq = rfftfreq(len(x_raw),d=1/sample_freq)
    fft = np.abs(rfft(x_raw))
    
    for i,j in zip(feat_divisions,feat_divisions[1:]):
        start = np.argmax(freq>i)
        stop = np.argmax(freq>j)
        features.append(np.max(fft[start:stop]))
        
    return features, feature_names
    

def gen_features4(x_raw):
    """
    Helper function to generate simple statistical features from a 150,000 row 'acoustic data' segment.
    x_raw must be a pandas series
    """
    
    feature_names = ['mean', 'std', 'max', 'min', 'skew',
                     'kurtosis','r10_std','r100_std','r1000_std', '1diff_mean',
                     '1diff_std','2diff_std','cumsum_max','num > q0.995',
                     ]

    features = [x_raw.mean(),
                x_raw.std(),
                x_raw.max(),
                x_raw.min(),
                x_raw.skew(),
                x_raw.kurtosis(),
                x_raw.rolling(10).mean().std(),
                x_raw.rolling(100).mean().std(),
                x_raw.rolling(1000).mean().std(),
                x_raw.diff().mean(),
                x_raw.diff().std(),
                x_raw.diff().diff().std(),
                (x_raw - x_raw.mean()).cumsum().abs().max(),
                num_spikes(x_raw),
            ]
    
    return features, feature_names


def gen_features5(x_raw,feat_divisions=[1e4,6e4,1.2e5,1.6e5,2e5,2.3e5,3e5],sample_freq = 4e6):
    """
    Helper function to generate simple statistical and fft features from a 150,000 row 'acoustic data' segment.
    x_raw must be a pandas series
    feat_divisions is the ranges of frequencies in Hz to store fourier coefficients for.
    sample_freq is the sampling frequency of the provided data. The default is 4MHz.
    
    returns the converted features and the names of the names of the features.
    """
    
    feature_names = ['mean', 'std', 'max', 'min', 'skew',
                     'kurtosis','r10_std','r100_std','r1000_std', '1diff_mean',
                     '1diff_std','2diff_std','cumsum_max','num > q0.995',
                     ]
    feature_names += ["fft_[{:},{:}]Hz".format(int(i),int(j)) for (i,j) in zip(feat_divisions,feat_divisions[1:])]


    features = [x_raw.mean(),
                x_raw.std(),
                x_raw.max(),
                x_raw.min(),
                x_raw.skew(),
                x_raw.kurtosis(),
                x_raw.rolling(10).mean().std(),
                x_raw.rolling(100).mean().std(),
                x_raw.rolling(1000).mean().std(),
                x_raw.diff().mean(),
                x_raw.diff().std(),
                x_raw.diff().diff().std(),
                (x_raw - x_raw.mean()).cumsum().abs().max(),
                num_spikes(x_raw),
            ]
    
    freq = rfftfreq(len(x_raw),d=1/sample_freq)
    fft = np.abs(rfft(x_raw))
    
    for i,j in zip(feat_divisions,feat_divisions[1:]):
        start = np.argmax(freq>i)
        stop = np.argmax(freq>j)
        features.append(np.max(fft[start:stop]))
    
    return features, feature_names



def num_spikes(x_raw):
    """
    Helper function to find the number of points where the absolute value of the acoustic data is greater 
    than the 0.995 quantile value. Ie it attempts to measure the number of 'spikes' in the data.
    
    x_raw must be a pandas series
    """
    x_raw_centred = np.abs(x_raw-x_raw.mean())
    return np.sum(x_raw_centred>x_raw_centred.quantile(0.995))


#------------------- Function to apply feature generation functions to the dataset -------------------------#
    
def gen_dim_reduced_dataset(k,gen_features = gen_features1,segment_size = 150000, overlap_ratio = 1):
    """
       Function to generate features with the k'th chunk of the dataset
       are generated with the 'gen_features' parameter
       
       segment_size = size of each data segment from which to generate features from
       overlap_ratio = amount of overlap between successive segments. If overlap = 1, there is no overlap. If 
       0 < overlap_ratio < 1, successive segments have an overlap of overlap_ratio * segment_size
    """
    
    # Read in k'th training chunk
    data = pd.read_csv(os.path.join("Data","train_chunk{0}.csv".format(k)))
    
    # Size of segment of acoustic data to be treated as an input
    segment_size = min(max(int(segment_size),1),data.shape[0])
    # Size of increment, ie the overlap between successive segments
    increment = min(max(int(overlap_ratio*segment_size),1),data.shape[0])
    
    # Create lists to keep track of data while iterating through segments in the training file.
    data_x = []
    data_y = []
    data_y_range = []
    
    # Iterate through segments in the training chunk
    for i in range(int(np.ceil(data.shape[0]/increment))):
        # get start and end indices of segment in the training chunk
        start = i*increment
        stop = min(start+segment_size, data.shape[0])
        
        # extract the raw data
        x_raw = data.loc[start:stop,'acoustic_data']
        y_raw = data.loc[start:stop,'time_to_failure']

        # Generate features from the raw data with the specified gen_features function
        xvals, feature_names = gen_features(x_raw)
        
        data_x.append(xvals)
        data_y.append(y_raw.iloc[-1])
        data_y_range.append(y_raw.max()-y_raw.min())
    
    # Collate the stored feature and target variables into dataframes
    x_df = pd.DataFrame(data_x, columns = feature_names)
    y_df = pd.DataFrame(data_y, columns = ['time_to_failure'])
    
    # store recovered data in a new csv file
    # [np.array(data_y_range)<1] is a boolean filter that 
    # avoids the labquake, spike regions of the training data
    pd.concat([x_df,y_df],axis=1)[np.array(data_y_range)<1].to_csv(os.path.join('temp','dim_reduced_train{}.csv'.format(k)))
    
    # Write a status test file 
    write_test_file(k,'segment_size = {}\n'.format(segment_size),
                    'increment/segment_size = {:.3f}\n'.format(increment/segment_size))



def gen_submission_features(test_folder=os.path.join("Data","test"),gen_features = gen_features1,fset=1, dest_folder = "Features"):
    """
       Function to generate features on the actual competition test.
       Features are generated with the 'gen_features' parameter
       fset = the number attached to the end of the final csv file with the features.
       
    """
    seg_ids = []
    data_x = []
    
    # Iterate through the submission test segments
    for csv in os.listdir(test_folder):
        # Save the name of the segment ids
        seg_ids.append(csv.split('.')[0])
        # read in the acoustic data
        x_raw = pd.read_csv(os.path.join(test_folder,csv))['acoustic_data']

        # Generate features from the raw data with the specified gen_features function
        xvals, feature_names = gen_features(x_raw)
        data_x.append(xvals)
    
    # Collate the stored feature and target variables into dataframes
    x_df1 = pd.DataFrame(seg_ids, columns = ['seg_id'])
    x_df2 = pd.DataFrame(data_x, columns = feature_names)
    
    # store recovered data in a new csv file
    pd.concat([x_df1,x_df2],axis=1).to_csv(os.path.join(dest_folder,"submission_features{}.csv".format(fset)))
    