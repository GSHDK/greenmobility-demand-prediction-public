from __future__ import print_function
from datetime import datetime, timedelta
import copy, torch
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import codecs


class DemandGridDataset(torch.utils.data.Dataset):
    """Demand grid dataset."""

    def __init__(self, csv_file, 
    	train=False, 
    	SEGMENT_MIN=30, 
    	vertical_lines = 10, 
    	horizontal_lines = 13, 
    	SEQ_LEN_SLICES=10, 
    	standardize=None, 
    	return_features=False,
        standardize_batchwise=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.            
            Note: The data is already transformed an will not accept further transformations
        """
        # Create the data
        self.sliding_elements=SEQ_LEN_SLICES
        self.SEGMENT_MIN=SEGMENT_MIN
        self.df_trip = pd.read_csv(csv_file)
        self.df_trip['startRentalCallSuccessfulTime'] = pd.to_datetime(self.df_trip['startRentalCallSuccessfulTime'])
        self.grid_slices = data_matrix(self.df_trip, vertical_lines, horizontal_lines, SEGMENT_MIN)
        
        # Standardization
        _slice_array = np.array([x[1] for x in self.grid_slices])
        self.mean_grid = np.mean(_slice_array,axis=0)
        self.std_grid = np.std(_slice_array,axis=0)
        
        del _slice_array

        # Can't do batch normalization and standardization at once
        assert not (standardize_batchwise and standardize)
        self.standardize_batchwise=standardize_batchwise

        # Test and validation requires a defined grid
        if standardize and not train:
                assert ((type(standardize)==list or type(standardize)==tuple) and len(standardize)==2)
                self.mean_grid=standardize[0]
                self.std_grid=standardize[1]
        self.standardize = standardize
        self.return_features = return_features
            
        
    def __len__(self):
        # Save an element for targets
        return max(len(self.grid_slices)-1-self.sliding_elements,1)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Grid
        grid_samples = [x[1] for x in self.grid_slices[idx:idx+self.sliding_elements]]
        
        # TODO: Implement a way of getting metadata about the segment. Return data and time 
        # Meta about sample
        self.df_trip.loc[set(self.grid_slices[idx][2])]
        # Time start of sample
        time_intervall = (self.grid_slices[idx][0],timedelta(minutes=self.SEGMENT_MIN))
        
        try:
            target = self.grid_slices[idx+self.sliding_elements+1][1]
            if self.standardize:
                # Standardize batch
                grid_samples = [standardize_grid(x, self.mean_grid, self.std_grid) for x in grid_samples]
                # Standardize target
                target = standardize_grid(target, self.mean_grid, self.std_grid)
            if self.standardize_batchwise:
                b_mean_grid = np.mean(grid_samples ,axis=0)
                b_std_grid =np.std(grid_samples ,axis=0)

                # Standardize batch
                grid_samples = [standardize_grid(x, b_mean_grid, b_std_grid) for x in grid_samples]
                # Standardize target
                target = standardize_grid(target, b_mean_grid, b_std_grid)


            grid_samples = torch.stack(list(map(torch.from_numpy, grid_samples)), 0)

            # Return the target time for feature engineering 
            if self.return_features:
                features = create_feature_dict_from_timestamp(self.grid_slices[idx+self.sliding_elements+1][0])
                return grid_samples, torch.from_numpy(target), features

            return grid_samples, torch.from_numpy(target)
        except IndexError as e:
            print(target)
            print(idx+self.sliding_elements+1)
            raise Exception("Went out of bounds in dataset classe - reconfigure item/label config",e)
            
    def zero_grid_elements(self, mask:np.array):
        raise NotImplementedError('Work in progress')
        assert mask.shape==self.mean_grid.shape
        total_demand = np.sum([x[1] for x in self.grid_slices],axis=0)
        demand_filter = total_demand[(total_demand<100) & (total_demand>0)]
        self.mean_grid[demand_filter]=0
        self.std_grid[demand_filter]=0
        for i in range(len(self.grid_slices)):
            self.grid_slices[i][1][demand_filter]=0

    def batchwise_std_inverse(self, idx, return_params=False):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Grid
        grid_samples = [x[1] for x in self.grid_slices[idx:idx+self.sliding_elements]]
        
        # TODO: Implement a way of getting metadata about the segment. Return data and time 
        # Meta about sample
        self.df_trip.loc[set(self.grid_slices[idx][2])]
        # Time start of sample
        time_intervall = (self.grid_slices[idx][0],timedelta(minutes=self.SEGMENT_MIN))
        
        b_mean_grid = np.mean(grid_samples ,axis=0)
        b_std_grid =np.std(grid_samples ,axis=0)

        data = self.__getitem__(idx)
        inp = [standardize_grid(x, b_mean_grid, b_std_grid, inverse=True) for x in data[0]]
        ta = standardize_grid(data[1],b_mean_grid,b_std_grid, inverse=True)
        if return_params:
            return inp, ta, b_mean_grid, b_std_grid
        return inp, ta





def standardize_grid(grid: np.array, mean_matrix: np.array, std_matrix: np.array, inverse=False):
    # Shape shape - only a single slice
    assert grid.shape == mean_matrix.shape == std_matrix.shape
    
    if inverse:
        return grid*(std_matrix+1e-7)+mean_matrix
    return (grid-mean_matrix)/(std_matrix+1e-7)


def create_feature_dict_from_timestamp(time):
    """
    For feature engineering. Measures porposed by filipe during meeting 2
    """
    def onehuttime(int_time, potential_values):
        _temp =np.zeros(potential_values)
        _temp[int_time]=1
        return _temp
    # Days since corona set to zero for before corona
    # Day of month off
    return {'date': time.strftime("%m-%d-%Y, %H:%M:%S"),
     'hour': onehuttime(time.hour,24),
     'dayofweek': onehuttime(time.dayofweek,7),
     'time_since_corona': max((time - datetime(2020,3,1)).days,0),
     'time_since_launch': (time - datetime(2019,1,27)).days,
     'day_of_month': onehuttime(time.day-1,31)
    }
    
def create_grid_from_rows(dataframe, list_of_indexes, grid_shape):
    grid=np.zeros(grid_shape)
    # Get non zero values of grid and insert their placement
    cnt=0
    for x in dataframe.loc[list_of_indexes,:].groupby(['col_start','row_start']).count()[['date']].iterrows():
        try:
            grid[x[0][1],x[0][0]]=x[1]['date']
        except:
            cnt+=1
            pass
    if cnt>0:
        print(cnt)
    return grid


def data_matrix(df_test: pd.DataFrame, vertical_lines: int, horizontal_lines: int, SEGMENT_MIN=30) -> pd.DataFrame:
    '''Tager data fra indlæsnings format og returnerer en matrix dimensionerne X_cor x Y_cor x 24.
    En time er for nærværende det nemmeste, da man kan bruge ".dt.hour" funktionaliteten i pandas'''
    # Start from first hour encountered
    cur_time= df_test.iloc[0]['startRentalCallSuccessfulTime'].replace(microsecond=0, second=0, minute=0)
    grid_slices=[]
    idx_set=set()
    itr = df_test.iterrows()
    x = next(itr)
    while itr:
        # Within the 30 minutes
        if cur_time >=(x[1]['startRentalCallSuccessfulTime']-timedelta(minutes=SEGMENT_MIN)):
            if x[0] not in idx_set:
                idx_set.add(x[0])
            try:
                x = next(itr)
            except:
                break
        # Moving to next time bin    
        else:
            # This could also just generate the grid slice
            idx_list=list(idx_set)
            idx_set.clear()
            _grid = create_grid_from_rows(df_test,idx_list, (vertical_lines+1,horizontal_lines+1))
            grid_slices.append((cur_time, _grid, idx_list))
            idx_list.clear()
            idx_list.append(x[0])
            cur_time=cur_time+timedelta(minutes=SEGMENT_MIN)
    return grid_slices

class MovingMNIST(data.Dataset):
    """`MovingMNIST <http://www.cs.toronto.edu/~nitish/unsupervised_video/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        split (int, optional): Train/test split size. Number defines how many samples
            belong to test set. 
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in an PIL
            image and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    urls = [
        'https://github.com/tychovdo/MovingMNIST/raw/master/mnist_test_seq.npy.gz'
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'moving_mnist_train.pt'
    test_file = 'moving_mnist_test.pt'

    def __init__(self, root, train=True, split=1000, transform=None, target_transform=None, max_norm=True, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.train = train  # training set or test set

        self.max_norm=max_norm

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.train_data = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
        else:
            self.test_data = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (seq, target) where sampled sequences are splitted into a seq
                    and target part
        """

        # need to iterate over time
        def _transform_time(data):
            new_data = None
            for i in range(data.size(0)):
                img = Image.fromarray(data[i].numpy(), mode='L')
                new_data = self.transform(img) if new_data is None else torch.cat([self.transform(img), new_data], dim=0)
            return new_data

        if self.train:
            seq, target = self.train_data[index, :10], self.train_data[index, 10:]
        else:
            seq, target = self.test_data[index, :10], self.test_data[index, 10:]

        if self.transform is not None:
            seq = _transform_time(seq)
        if self.target_transform is not None:
            target = _transform_time(target)
        if self.max_norm:
            # rescale max value
            seq = seq/255.0
            target = target/255.0

        return seq/1.0, target/1.0

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):
        """Download the Moving MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        training_set = torch.from_numpy(
            np.load(os.path.join(self.root, self.raw_folder, 'mnist_test_seq.npy')).swapaxes(0, 1)[:-self.split]
        )
        test_set = torch.from_numpy(
            np.load(os.path.join(self.root, self.raw_folder, 'mnist_test_seq.npy')).swapaxes(0, 1)[-self.split:]
        )

        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')