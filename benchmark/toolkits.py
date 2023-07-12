"""
DISTRIBUTION OF DATASET
-----------------------------------------------------------------------------------
balance:
    iid:            0 : identical and independent distributions of the dataset among clients
    label skew:     1 Quantity:  each party owns data samples of a fixed number of labels.
                    2 Dirichlet: each party is allocated a proportion of the samples of each label according to Dirichlet distribution.
                    3 Shard: each party is allocated the same numbers of shards that is sorted by the labels of the data
-----------------------------------------------------------------------------------
depends on partitions:
    feature skew:   4 Noise: each party owns data samples of a fixed number of labels.
                    5 ID: For Shakespeare\FEMNIST, we divide and assign the writers (and their characters) into each party randomly and equally.
-----------------------------------------------------------------------------------
imbalance:
    iid:            6 Vol: only the vol of local dataset varies.
    niid:           7 Vol: for generating synthetic data
"""
import torch
import ujson
import numpy as np
import os.path
import random
import urllib
import zipfile
import os
import ssl
from torch.utils.data import Dataset, DataLoader
import torch
ssl._create_default_https_context = ssl._create_unverified_context
import importlib
from sklearn.utils import shuffle
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def set_random_seed(seed=0):
    """Set random seed"""
    random.seed(3 + seed)
    np.random.seed(97 + seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def download_from_url(url= None, filepath = '.'):
    """Download dataset from url to filepath."""
    if url: urllib.request.urlretrieve(url, filepath)
    return filepath

def extract_from_zip(src_path, target_path):
    """Unzip the .zip file (src_path) to target_path"""
    f = zipfile.ZipFile(src_path)
    f.extractall(target_path)
    targets = f.namelist()
    f.close()
    return [os.path.join(target_path, tar) for tar in targets]




class BasicTaskGen:
    _TYPE_DIST = {
        0: 'iid',
        1: 'label_skew_quantity',
        2: 'label_skew_dirichlet',
        3: 'label_skew_shard',
        4: 'feature_skew_noise',
        5: 'feature_skew_id',
        6: 'iid_volumn_skew',
        7: 'niid_volumn_skew',
        8: 'concept skew',
        9: 'concept and feature skew and balance',
        10: 'concept and feature skew and imbalance',
    }
    _TYPE_DATASET = ['2DImage', '3DImage', 'Text', 'Sequential', 'Graph', 'Tabular']

    def __init__(self, benchmark, dist_id, skewness, rawdata_path, seed=0):
        self.benchmark = benchmark
        self.rootpath = './fedtask'
        if not os.path.exists(self.rootpath):
            os.mkdir(self.rootpath)

        self.rawdata_path = rawdata_path
        self.dist_id = dist_id
        self.dist_name = self._TYPE_DIST[dist_id]
        self.skewness = 0 if dist_id==0 else skewness
        self.num_clients = -1
        self.seed = seed
        set_random_seed(self.seed)

    def run(self):
        """The whole process to generate federated task. """
        pass

    def load_data(self):
        """Download and load dataset into memory."""
        pass

    def partition(self):
        """Partition the data according to 'dist' and 'skewness'"""
        pass

    def save_data(self):
        """Save the federated dataset to the task_path/data.
        This algorithm should be implemented as the way to read
        data from disk that is defined by DataReader.read_data()
        """
        pass

    def save_info(self):
        """Save the task infomation to the .json file stored in taskpath"""
        pass

    def get_taskname(self):
        """Create task name and return it."""
        taskname = '_'.join([self.benchmark, 'cnum' +  str(self.num_clients), 'dist' + str(self.dist_id), 'skew' + str(self.skewness).replace(" ", ""), 'seed'+str(self.seed)])
        return taskname

    def get_client_names(self):
        k = str(len(str(self.num_clients)))
        return [('Client{:0>' + k + 'd}').format(i) for i in range(self.num_clients)]

    def create_task_directories(self):
        """Create the directories of the task."""
        taskname = self.get_taskname()
        taskpath = os.path.join(self.rootpath, taskname)
        os.mkdir(taskpath)
        os.mkdir(os.path.join(taskpath, 'record'))

    def _check_task_exist(self):
        """Check whether the task already exists."""
        taskname = self.get_taskname()
        print(taskname)
        return os.path.exists(os.path.join(self.rootpath, taskname))

class DefaultTaskGen(BasicTaskGen):
    def __init__(self, benchmark, dist_id, skewness, rawdata_path, num_clients=1, minvol=10, seed=0):
        super(DefaultTaskGen, self).__init__(benchmark, dist_id, skewness, rawdata_path, seed)
        self.minvol=minvol
        self.num_classes = -1
        self.train_data = None
        self.test_data = None
        self.num_clients = num_clients
        self.cnames = self.get_client_names()
        self.taskname = self.get_taskname()
        self.taskpath = os.path.join(self.rootpath, self.taskname)
        self.save_data = self.XYData_to_json
        self.label_after_sort_case3 = None
        self.datasrc = {
            'lib': None,
            'class_name': None,
            'args':[]
        }

    def run(self):
        """ Generate federated task"""
        # check if the task exists
        if not self._check_task_exist():
            self.create_task_directories()
        else:
            print("Task Already Exists.")
            return
        # read raw_data into self.train_data and self.test_data
        print('-----------------------------------------------------')
        print('Loading...')
        self.load_data()
        print('Done.')
        # partition data and hold-out for each local dataset
        print('-----------------------------------------------------')
        print('Partitioning data...')
        labels=[]
        local_datas = self.partition()
        # print("Local data", local_datas)
        # ? Đoạn này chỉ đơn thuần là split data, không ảnh hưởng
        # train_cidxs, valid_cidxs = self.local_holdout(local_datas, rate=0.8, shuffle=True)
        train_cidxs, valid_cidxs = self.local_holdout_2(local_datas, rate=0.8, shuffle=True)
        
        print('Done.')
        # save task infomation as .json file and the federated dataset
        print('-----------------------------------------------------')
        print('Saving data...')
        self.save_info()
        self.save_data(train_cidxs, valid_cidxs)
        print('Done.')
        return

    def load_data(self):
        """ load and pre-process the raw data"""
        return

    def divide_data_balance(self, num_local_class=10, i_seed=0):
        torch.manual_seed(i_seed)

        config_division = {}  # Count of the classes for division
        config_class = {}  # Configuration of class distribution in clients
        config_data = {}  # Configuration of data indexes for each class : Config_data[cls] = [0, []] | pointer and indexes
        dataset = self.train_data
        total_data_points = len(dataset)

        # Count the occurrences of each label in the dataset
        label_counts = [0] * 100
        for _, label in dataset:
            label_counts[label] += 1

        # Print the label and corresponding count
        for label, count in enumerate(label_counts):
            print(f"Label {label}: {count} data points")
        for i in range(self.num_clients):
            config_class['f_{0:05d}'.format(i)] = []
            for j in range(num_local_class):
                cls = (i+j) % self.num_classes
                if cls not in config_division:
                    config_division[cls] = 1
                    config_data[cls] = [0, []]

                else:
                    config_division[cls] += 1
                config_class['f_{0:05d}'.format(i)].append(cls)
        dpairs = [[did, self.train_data[did][-1]] for did in range(len(self.train_data))]
        train_targets = torch.tensor([p[1] for p in dpairs])

        for cls in config_division.keys():
            indexes = torch.nonzero(train_targets == cls)
            num_datapoint = indexes.shape[0]
            indexes = indexes[torch.randperm(num_datapoint)]
            num_partition = num_datapoint // config_division[cls]

            for i_partition in range(config_division[cls]):
                if i_partition == config_division[cls] - 1:
                    config_data[cls][1].append(indexes[i_partition * num_partition:])
                else:
                    config_data[cls][1].append(indexes[i_partition * num_partition: (i_partition + 1) * num_partition])

        local_datas_in_def = [[] for i in range(self.num_clients)]
        index_client = 0
        for user in tqdm(config_class.keys()):
            user_data_indexes = torch.tensor([])
            for cls in config_class[user]:
                # !config_data[cls][0] auto là số 0 :| , [1] là các array chia cho data đó
                user_data_index = config_data[cls][1][config_data[cls][0]]
                print(f'user: {user} cls: {cls} user_data_index: {len(user_data_index)}')
                user_data_indexes = torch.cat((user_data_indexes, user_data_index))
                config_data[cls][0] += 1
                # print(len(user_data_indexes))
            indexs_of_user = [int(i[0]) for i in user_data_indexes.tolist()]
            print(f'{user}  - {len(indexs_of_user)}')
            local_datas_in_def[index_client] = indexs_of_user
            index_client+=1
        return local_datas_in_def


    def divide_data_balance_anhtn(self, num_local_class=10, i_seed=0):
        torch.manual_seed(i_seed)

        config_division = {}  # Count of the classes for division
        config_class = {}  # Configuration of class distribution in clients
        config_data = {}  # Configuration of data indexes for each class : Config_data[cls] = [0, []] | pointer and indexes
        dataset = self.train_data
        total_data_points = len(dataset)

        # Count the occurrences of each label in the dataset
        label_counts = [0] * 100
        for _, label in dataset:
            label_counts[label] += 1

        # Print the label and corresponding count
        for label, count in enumerate(label_counts):
            print(f"Label {label}: {count} data points")
        for i in range(self.num_clients):
            config_class['f_{0:05d}'.format(i)] = []
            for j in range(num_local_class):
                cls = (i+j) % self.num_classes
                if cls not in config_division:
                    config_division[cls] = 1
                    config_data[cls] = [0, []]

                else:
                    config_division[cls] += 1
                config_class['f_{0:05d}'.format(i)].append(cls)
        dpairs = [[did, self.train_data[did][-1]] for did in range(len(self.train_data))]
        train_targets = torch.tensor([p[1] for p in dpairs])

        for cls in config_division.keys():
            indexes = torch.nonzero(train_targets == cls)
            num_datapoint = indexes.shape[0]
            indexes = indexes[torch.randperm(num_datapoint)]
            num_partition = num_datapoint // config_division[cls]

            for i_partition in range(config_division[cls]):
                if i_partition == config_division[cls] - 1:
                    config_data[cls][1].append(indexes[i_partition * num_partition:])
                else:
                    config_data[cls][1].append(indexes[i_partition * num_partition: (i_partition + 1) * num_partition])

        local_datas_in_def = [[] for i in range(self.num_clients)]
        index_client = 0
        for user in tqdm(config_class.keys()):
            user_data_indexes = torch.tensor([])
            for cls in config_class[user]:
                # !config_data[cls][0] auto là số 0 :| , [1] là các array chia cho data đó
                user_data_index = config_data[cls][1][config_data[cls][0]]
                print(f'user: {user} cls: {cls} user_data_index: {len(user_data_index)}')
                user_data_indexes = torch.cat((user_data_indexes, user_data_index))
                config_data[cls][0] += 1
                # print(len(user_data_indexes))
            indexs_of_user = [int(i[0]) for i in user_data_indexes.tolist()]
            print(f'{user}  - {len(indexs_of_user)}')
            local_datas_in_def[index_client] = indexs_of_user
            index_client+=1

        return local_datas_in_def



    def partition(self):
        # Partition self.train_data according to the delimiter and return indexes of data owned by each client as [c1data_idxs, ...] where the type of each element is list(int)
        if self.dist_id == 0:
            """IID"""
            d_idxs = np.random.permutation(len(self.train_data))
            local_datas = np.array_split(d_idxs, self.num_clients)

        elif self.dist_id == 1:
            """label_skew_quantity"""
            # print("skewness", self.skewness)
            self.skewness = min(max(0, self.skewness),1.0)
            print("skewness", self.skewness)
            dpairs = [[did, self.train_data[did][-1]] for did in range(len(self.train_data))]
            num = max(int((1-self.skewness)*self.num_classes), 1)
            # print("Num: ", num)
            K = self.num_classes
            local_datas = [[] for _ in range(self.num_clients)]
            if num == K:
                for k in range(K):
                    idx_k = [p[0] for p in dpairs if p[1]==k]
                    np.random.shuffle(idx_k)
                    split = np.array_split(idx_k, self.num_clients)
                    for cid in range(self.num_clients):
                        local_datas[cid].extend(split[cid].tolist())
            else:
                times = [0 for _ in range(self.num_classes)]
                contain = []
                for i in range(self.num_clients):
                    current = [i % K]
                    times[i % K] += 1
                    j = 1
                    while (j < num):
                        ind = random.randint(0, K - 1)
                        if (ind not in current):
                            j = j + 1
                            current.append(ind)
                            times[ind] += 1
                    contain.append(current)
                for k in range(K):
                    idx_k = [p[0] for p in dpairs if p[1]==k]
                    np.random.shuffle(idx_k)
                    split = np.array_split(idx_k, times[k])
                    ids = 0
                    for cid in range(self.num_clients):
                        if k in contain[cid]:
                            local_datas[cid].extend(split[ids].tolist())
                            ids += 1

        elif self.dist_id == 2:
            """label_skew_dirichlet"""
            min_size = 0
            dpairs = [[did, self.train_data[did][-1]] for did in range(len(self.train_data))]
            local_datas = [[] for _ in range(self.num_clients)]
            while min_size < self.minvol:
                idx_batch = [[] for i in range(self.num_clients)]
                for k in range(self.num_classes):
                    idx_k = [p[0] for p in dpairs if p[1]==k]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(np.repeat(self.skewness, self.num_clients))
                    ## Balance
                    proportions = np.array([p * (len(idx_j) < len(self.train_data)/ self.num_clients) for p, idx_j in zip(proportions, idx_batch)])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                    min_size = min([len(idx_j) for idx_j in idx_batch])
            for j in range(self.num_clients):
                np.random.shuffle(idx_batch[j])
                local_datas[j].extend(idx_batch[j])

        elif self.dist_id == 3:
            """label_skew_shard"""
            # Get index and label in train data
            dpairs = [[did, self.train_data[did][-1]] for did in range(len(self.train_data))]
            self.skewness = min(max(0, self.skewness), 1.0)
            num_shards = max(int((1 - self.skewness) * self.num_classes * 2), 1)
            client_datasize = int(len(self.train_data) / self.num_clients)
            all_idxs = [i for i in range(len(self.train_data))]
            z = zip([p[1] for p in dpairs], all_idxs)
            z = sorted(z)
            labels, all_idxs = zip(*z)
            self.label_after_sort_case3 = labels
            shardsize = int(client_datasize / num_shards)
            idxs_shard = range(int(self.num_clients * num_shards))
            local_datas = [[] for i in range(self.num_clients)]
            for i in range(self.num_clients):
                # if i ==22:
                #     import pdb; pdb.set_trace()
                rand_set = set(np.random.choice(idxs_shard, num_shards, replace=False))
                idxs_shard = list(set(idxs_shard) - rand_set)
                # for rand in rand_set:
                #     local_datas[i].extend(all_idxs[rand * shardsize:(rand + 1) * shardsize])
                for rand in rand_set:
                    sorted_indices = all_idxs[rand * shardsize:(rand + 1) * shardsize]
                    original_indices = [all_idxs[idx] for idx in sorted_indices]
                    local_datas[i].extend(original_indices)

            for i in range(self.num_clients):
                # ... Rest of the code in the loop ...

                # Calculating distinct labels for client i
                client_data_indices = local_datas[i]
                client_labels = [self.train_data[did][-1] for did in client_data_indices]
                distinct_labels = set(client_labels)

                # Log the distinct labels for the client
                print(f'After running client {i}, distinct labels: {distinct_labels}')
                    

        elif self.dist_id == 4:
            pass

        elif self.dist_id == 5:
            """feature_skew_id"""
            if not isinstance(self.train_data, TupleDataset):
                raise RuntimeError("Support for dist_id=5 only after setting the type of self.train_data is TupleDataset")
            Xs, IDs, Ys = self.train_data.tolist()
            self.num_clients = len(set(IDs))
            local_datas = [[] for _ in range(self.num_clients)]
            for did in range(len(IDs)):
                local_datas[IDs[did]].append(did)

        elif self.dist_id == 6:
            minv = 0
            d_idxs = np.random.permutation(len(self.train_data))
            while minv < self.minvol:
                proportions = np.random.dirichlet(np.repeat(self.skewness, self.num_clients))
                proportions = proportions / proportions.sum()
                minv = np.min(proportions * len(self.train_data))
            proportions = (np.cumsum(proportions) * len(d_idxs)).astype(int)[:-1]
            local_datas  = np.split(d_idxs, proportions)



        elif self.dist_id == 7:
           local_datas = self.divide_data_balance(num_local_class=2)
        elif self.dist_id == 8:
           local_datas = self.divide_data_balance_anhtn(num_local_class=2)
                    
        
        return local_datas

    def local_holdout(self, local_datas, rate=0.8, shuffle=False):
        """split each local dataset into train data and valid data according the rate."""
        train_cidxs = []
        valid_cidxs = []
        # Bản chất là từng client
        for local_data in local_datas:
            if shuffle:
                np.random.shuffle(local_data)
            k = int(len(local_data) * rate)
            train_cidxs.append(local_data[:k])
            valid_cidxs.append(local_data[k:])
        return train_cidxs, valid_cidxs


    def local_holdout_2(self, local_datas, rate=0.8, shuffle=False):
        """split each local dataset into train data and valid data according the rate."""
        train_cidxs = []
        valid_cidxs = []
        # Bản chất là từng client
        for local_data in local_datas:
            if shuffle:
                np.random.shuffle(local_data)
            k = int(len(local_data) * rate)
            x_train_idx, x_test = train_test_split(local_data, train_size=rate)
            train_cidxs.append(x_train_idx)
            valid_cidxs.append(x_test)
        return train_cidxs, valid_cidxs


    def save_info(self):
        info = {
            'benchmark': self.benchmark,  # name of the dataset
            'dist': self.dist_id,  # type of the partition way
            'skewness': self.skewness,  # hyper-parameter for controlling the degree of niid
            'num-clients': self.num_clients,  # numbers of all the clients
        }
        # save info.json
        with open(os.path.join(self.taskpath, 'info.json'), 'w') as outf:
            ujson.dump(info, outf)

    def convert_data_for_saving(self):
        """Convert self.train_data and self.test_data to list that can be stored as .json file and the converted dataset={'x':[], 'y':[]}"""
        pass

    def XYData_to_json(self, train_cidxs, valid_cidxs):
        self.convert_data_for_saving()
        # save federated dataset
        feddata = {
            'store': 'XY',
            'client_names': self.cnames,
            'dtest': self.test_data

        }
        for cid in range(self.num_clients):
            feddata[self.cnames[cid]] = {
                'dtrain':{
                    'x':[self.train_data['x'][did] for did in train_cidxs[cid]], 'y':[self.train_data['y'][did] for did in train_cidxs[cid]]
                },
                'dvalid':{
                    'x':[self.train_data['x'][did] for did in valid_cidxs[cid]], 'y':[self.train_data['y'][did] for did in valid_cidxs[cid]]
                }
            }

        with open(os.path.join(self.taskpath, 'data.json'), 'w') as outf:
            ujson.dump(feddata, outf)
        return

    def IDXData_to_json(self, train_cidxs, valid_cidxs):
        if self.datasrc ==None:
            raise RuntimeError("Attr datasrc not Found. Please define it in __init__() before calling IndexData_to_json")
        feddata = {
            'store': 'IDX',
            'client_names': self.cnames,
            'dtest': [i for i in range(len(self.test_data))],
            'datasrc': self.datasrc
        }
        for cid in range(self.num_clients):
            feddata[self.cnames[cid]] = {
                'dtrain': train_cidxs[cid],
                'dvalid': valid_cidxs[cid]
            }
        with open(os.path.join(self.taskpath, 'data.json'), 'w') as outf:
            ujson.dump(feddata, outf)
        return

class BasicTaskCalculator:

    _OPTIM = None

    def __init__(self, device):
        self.device = device
        self.lossfunc = None
        self.DataLoader = None

    def data_to_device(self, data):
        raise NotImplementedError

    def get_loss(self):
        raise NotImplementedError

    def get_evaluation(self):
        raise NotImplementedError

    def get_data_loader(self, data, batch_size = 64):
        return NotImplementedError

    def test(self):
        raise NotImplementedError

    def get_optimizer(self, name="sgd", model=None, lr=0.1, weight_decay=0, momentum=0):
        # if self._OPTIM == None:
        #     raise RuntimeError("TaskCalculator._OPTIM Not Initialized.")
        if name.lower() == 'sgd':
            return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif name.lower() == 'adam':
            return torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay, amsgrad=True)
        else:
            raise RuntimeError("Invalid Optimizer.")

    @classmethod
    def setOP(cls, OP):
        cls._OPTIM = OP

class ClassifyCalculator(BasicTaskCalculator):
    def __init__(self, device):
        super(ClassifyCalculator, self).__init__(device)
        self.lossfunc = torch.nn.CrossEntropyLoss()
        self.DataLoader = DataLoader

    def get_loss(self, model, data, device=None):
        tdata = self.data_to_device(data, device)
        input, target = tdata[0], tdata[1].type(torch.LongTensor)
        target = target.to(self.device)

        outputs = model(input)
        loss = self.lossfunc(outputs, target)
        return loss

    @torch.no_grad()
    def get_evaluation(self, model, data):
        tdata = self.data_to_device(data)
        outputs = model(tdata)
        y_pred = outputs.data.max(1, keepdim=True)[1]
        correct = y_pred.eq(tdata[1].data.view_as(y_pred)).long().cpu().sum()
        return (1.0 * correct / len(tdata[1])).item()

    @torch.no_grad()
    def test(self, model, data, device=None):
        """Metric = Accuracy"""
        tdata = self.data_to_device(data, device)
        # print("Data shape",tdata[0].shape, tdata[1].shape)
        input, target = tdata[0], tdata[1].type(torch.LongTensor)
        target = target.to(self.device)
        # input = input.squeeze(1)
        # print(input.shape)
        # input = torch.flatten(input, start_dim = 1, end_dim = 2)
        # print(input.shape)
        model = model.to(device)
        outputs = model(input)
        # print(outputs.dtype, target.dtype)
        loss = self.lossfunc(outputs, target)
        y_pred = outputs.data.max(1, keepdim=True)[1]
        correct = y_pred.eq(tdata[1].data.view_as(y_pred)).long().cpu().sum()
        return (1.0 * correct / len(tdata[1])).item(), loss.item()

    def data_to_device(self, data, device=None):
        if device is None:
            return data[0].to(self.device), data[1].to(self.device)
        else:
            return data[0].to(device), data[1].to(device)

    def get_data_loader(self, dataset, batch_size=64, shuffle=True, droplast=False):
        if self.DataLoader == None:
            raise NotImplementedError("DataLoader Not Found.")
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=droplast)

class BasicTaskReader:
    def __init__(self, taskpath=''):
        self.taskpath = taskpath

    def read_data(self):
        """
            Reading the spilted dataset from disk files and loading data into the class 'LocalDataset'.
            This algorithm should read three types of data from the processed task:
                train_sets = [client1_train_data, ...] where each item is an instance of 'LocalDataset'
                valid_sets = [client1_valid_data, ...] where each item is an instance of 'LocalDataset'
                test_set = test_dataset
            Return train_sets, valid_sets, test_set, client_names
        """
        pass

class XYTaskReader(BasicTaskReader):
    def __init__(self, taskpath=''):
        super(XYTaskReader, self).__init__(taskpath)

    def read_data(self):
        with open(os.path.join(self.taskpath, 'data.json'), 'r') as inf:
            feddata = ujson.load(inf)
        # print(feddata.keys())
        # print(feddata['store'])
        test_data = XYDataset(feddata['dtest']['x'], feddata['dtest']['y'])
        train_datas = [XYDataset(feddata[name]['dtrain']['x'], feddata[name]['dtrain']['y']) for name in feddata['client_names']]
        valid_datas = [XYDataset(feddata[name]['dvalid']['x'], feddata[name]['dvalid']['y']) for name in feddata['client_names']]
        return train_datas, valid_datas, test_data, feddata['client_names']


class WholeTaskReader(BasicTaskReader):
    def __init__(self, taskpath=''):
        super(WholeTaskReader, self).__init__(taskpath)

    def read_data(self):
        with open(os.path.join(self.taskpath, 'data.json'), 'r') as inf:
            feddata = ujson.load(inf)
        # print(np.array(feddata['dtest']['x']).shape, np.array(feddata['dtest']['y']).shape)
        # print(feddata['store'])
        train_and_valid_X = []
        # valid_X = []
        train_and_valid_Y = []
        # valid_Y = []
        for client_name in feddata['client_names']:
            client_train_X, client_train_Y = feddata[client_name]['dtrain']['x'],  feddata[client_name]['dtrain']['y']
            # print("Client train x ", client_train_X)
            client_valid_X, client_valid_Y = feddata[client_name]['dvalid']['x'],  feddata[client_name]['dvalid']['y']
            # print(np.array(client_train_X).shape, np.array(client_train_Y).shape, np.array(client_valid_X).shape)
            train_and_valid_X.append(np.array(client_train_X))
            train_and_valid_Y.append(np.array(client_train_Y))
            train_and_valid_X.append(np.array(client_valid_X))
            train_and_valid_Y.append(np.array(client_valid_Y))
        
        train_and_valid_X, train_and_valid_Y = np.concatenate(train_and_valid_X, axis=0),  np.concatenate(train_and_valid_Y, axis=0)
        # print(train_X.shape, train_Y.shape, valid_X.shape)
        train_and_valid_data = XYDataset(train_and_valid_X, train_and_valid_Y)
        test_data = XYDataset(feddata['dtest']['x'], feddata['dtest']['y'])
        # train_datas = [XYDataset(feddata[name]['dtrain']['x'], feddata[name]['dtrain']['y']) for name in feddata['client_names']]
        # valid_datas = [XYDataset(feddata[name]['dvalid']['x'], feddata[name]['dvalid']['y']) for name in feddata['client_names']]
        return train_and_valid_data, test_data

class IDXTaskReader(BasicTaskReader):
    def __init__(self, taskpath=''):
        super(IDXTaskReader, self).__init__(taskpath)

    def read_data(self):
        with open(os.path.join(self.taskpath, 'data.json'), 'r') as inf:
            feddata = ujson.load(inf)
        DS = getattr(importlib.import_module(feddata['datasrc']['lib']), feddata['datasrc']['class_name'])
        arg_strings = '(' + ','.join(feddata['datasrc']['args'])
        train_args = arg_strings + ', train=True)'
        test_args = arg_strings + ', train=False)'
        DS.SET_DATA(eval(feddata['datasrc']['class_name'] + train_args))
        DS.SET_DATA(eval(feddata['datasrc']['class_name'] + test_args), key='TEST')
        test_data = IDXDataset(feddata['dtest'], key='TEST')
        train_datas = [IDXDataset(feddata[name]['dtrain']) for name in feddata['client_names']]
        valid_datas = [IDXDataset(feddata[name]['dvalid']) for name in feddata['client_names']]
        return train_datas, valid_datas, test_data, feddata['client_names']

class XYDataset(Dataset):
    def __init__(self, X=[], Y=[], client_name=None, totensor = True):
        """ Init Dataset with pairs of features and labels/annotations.
        XYDataset transforms data that is list\array into tensor.
        The data is already loaded into memory before passing into XYDataset.__init__()
        and thus is only suitable for benchmarks with small size (e.g. CIFAR10, MNIST)
        Args:
            X: a list of features
            Y: a list of labels with the same length of X
        """
        # print("XYDATASET", X,Y)
        if not self._check_equal_length(X, Y):
            raise RuntimeError("Different length of Y with X.")
        if totensor:
            try:
                self.X = torch.Tensor(X)
                self.Y = torch.Tensor(Y)
            except:
                raise RuntimeError("Failed to convert input into torch.Tensor.")
        else:
            self.X = X
            self.Y = Y
        self.client_name = client_name
        try:
            self.all_labels = list(set(self.tolist()[1]))
        except Exception as e:
            print("Error:", e)
            import pdb; pdb.set_trace()

        self.X, self.Y = shuffle(self.X,self.Y, random_state = 0)
        # self.X, self.Y = self.X, self.Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, item):
        return self.X[item], self.Y[item]
    
    def get_data(self):
        return self.X, self.Y

    def tolist(self):

        if not isinstance(self.X, torch.Tensor):
            return self.X, self.Y
        return self.X.tolist(), self.Y.tolist()

    def _check_equal_length(self, X, Y):
        return len(X)==len(Y)

    def get_all_labels(self):
        return self.all_labels

class IDXDataset(Dataset):
    # The source dataset that can be indexed by IDXDataset
    _DATA = {'TRAIN': None,'TEST': None}

    def __init__(self, idxs, key='TRAIN'):
        """Init dataset with 'src_data' and a list of indexes that are used to position data in 'src_data'"""
        if not isinstance(idxs, list):
            raise RuntimeError("Invalid Indexes")
        self.idxs = idxs
        self.key = key

    @classmethod
    def SET_DATA(cls, dataset, key = 'TRAIN'):
        cls._DATA[key] = dataset

    @classmethod
    def ADD_KEY_TO_DATA(cls, key, value = None):
        if key==None:
            raise RuntimeError("Empty key when calling class algorithm IDXData.ADD_KEY_TO_DATA")
        cls._DATA[key]=value

    def __getitem__(self, item):
        idx = self.idxs[item]
        return self._DATA[self.key][idx]

class TupleDataset(Dataset):
    def __init__(self, X1=[], X2=[], Y=[], totensor=True):
        if totensor:
            try:
                self.X1 = torch.tensor(X1)
                self.X2 = torch.tensor(X2)
                self.Y = torch.tensor(Y)
            except:
                raise RuntimeError("Failed to convert input into torch.Tensor.")
        else:
            self.X1 = X1
            self.X2 = X2
            self.Y = Y

    def __getitem__(self, item):
        return self.X1[item], self.X2[item], self.Y[item]

    def __len__(self):
        return len(self.Y)

    def tolist(self):
        if not isinstance(self.X1, torch.Tensor):
            return self.X1, self.X2, self.Y
        return self.X1.tolist(), self.X2.tolist(), self.Y.tolist()
