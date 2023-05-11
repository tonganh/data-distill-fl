import random
from utils import fmodule
import sys
sys.path.append('..')
from .fedbase_mobile  import BasicCloudServer, BasicEdgeServer, BasicMobileClient
from benchmark.toolkits import XYDataset
import copy
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from main_mobile import logger
import os
from tqdm import tqdm
from multiprocessing import Pool as ThreadPool


class CloudServer(BasicCloudServer):
    def __init__(self, option, model ,train_and_valid_data,test_data = None, clients = []):
        super(CloudServer, self).__init__( option, model,train_and_valid_data,test_data, clients )
        self.initialize()


    def run(self):
        """
        Start the federated learning symtem where the global model is trained iteratively.
        """
        logger.time_start('Total Time Cost')
        for round in range(self.num_rounds+1):
            print("--------------Round {}--------------".format(round))
            logger.time_start('Time Cost')

            # federated train
            self.iterate(round)
            # decay learning rate
            self.global_lr_scheduler(round)

            logger.time_end('Time Cost')
            if logger.check_if_log(round, self.eval_interval): logger.log(self)

        print("=================End==================")
        logger.time_end('Total Time Cost')
        # save results as .json file
        # logger.save(os.path.join('fedtask', self.option['task'], 'record', flw.output_filename(self.option, self)))

    def iterate(self, t):
        """
        The standard iteration of each federated round that contains three
        necessary procedure in FL: client selection, communication and model aggregation.
        :param
            t: the number of current round
        """
        # sample clients: MD sampling as default but with replacement=False
        # print("Iterating")
        self.global_update_location()
        # print("Done updating location")
        self.update_client_list()
        # print("Done updating client_list")
        self.assign_client_to_server()
        # print("Done assigning client to sercer")

        self.selected_clients = self.sample()
        # print("Selected clients", self.selected_clients)
        # print("Done sampling")
        # training
        models, train_losses = self.communicate(self.selected_clients)
        # print("Done a training step")
        # check whether all the clients have dropped out, because the dropped clients will be deleted from self.selected_clients
        if not self.selected_clients: return
        # aggregate: pk = 1/K as default where K=len(selected_clients)
        sum_datavol = sum( [client.datavol for client in self.selected_clients])
        agg_weights = [client.datavol /sum_datavol for client in self.selected_clients]
        self.model = self.aggregate(models, p = agg_weights)


    

    def communicate(self, selected_clients):
        """
        The whole simulating communication procedure with the selected clients.
        This part supports for simulating the client dropping out.
        :param
            selected_clients: the clients to communicate with
        :return
            :the unpacked response from clients that is created ny self.unpack()
        """
        packages_received_from_clients = []
        if self.num_threads <= 1:
            # computing iteratively
            print("Computing iteratively")
            for client_id in tqdm(selected_clients):
                response_from_client_id = self.communicate_with(client_id)
                packages_received_from_clients.append(response_from_client_id)
    
        else:
            # computing in parallel
            print("Computing in parallel")
            pool = ThreadPool(min(self.num_threads, len(selected_clients)))
            packages_received_from_clients = pool.map(self.communicate_with, selected_clients)
            pool.close()
            pool.join()
        # count the clients not dropping
        self.selected_clients = [selected_clients[i] for i in range(len(selected_clients)) if packages_received_from_clients[i]]
        packages_received_from_clients = [pi for pi in packages_received_from_clients if pi]
        return self.unpack(packages_received_from_clients)

    def communicate_with(self, client):
        """
        Pack the information that is needed for client_id to improve the global model
        :param
            client: the client to communicate with
        :return
            client_package: the reply from the client and will be 'None' if losing connection
        """
        # package the necessary information
        svr_pkg = self.pack()

        # listen for the client's response and return None if the client drops out
        # if self.clients[client_id].is_drop(): return None
        reply = client.reply(svr_pkg)
        return reply

    def pack(self):
        """
        Pack the necessary information for the client's local training.
        Any operations of compression or encryption should be done here.
        :param
            client_id: the id of the client to communicate with
        :return
            a dict that only contains the global model as default.
        """
        return {
            "model" : copy.deepcopy(self.model),
        }

    def sample(self):
        """Sample the clients.
        :param
            replacement: sample with replacement or not
        :return
            a list of the ids of the selected clients
        """
        # print("Sampling selected clients")
        all_clients = [cid for cid in range(self.num_clients)]
        # print("Done all clients")
        selected_clients = []
        # collect all the active clients at this round and wait for at least one client is active and
        active_clients = []
        active_clients = self.clients
        self.clients_per_round = self.current_num_clients
        # while(len(active_clients)<1):
        #     active_clients = [cid for cid in range(self.num_clients) if self.clients[cid].is_active()]
        # print("DOne collect all the active clients")
        # sample clients
        if self.sample_option == 'active':
            # select all the active clients without sampling
            selected_clients = active_clients
        if self.sample_option == 'uniform':
            # original sample proposed by fedavg
            selected_clients = list(np.random.choice(active_clients, self.clients_per_round, replace=False))
        elif self.sample_option =='md':
            # the default setting that is introduced by FedProx
            selected_clients = list(np.random.choice(all_clients, self.clients_per_round, replace=True, p=[nk / self.data_vol for nk in self.client_vols]))
        # drop the selected but inactive clients
        selected_clients = list(set(active_clients).intersection(selected_clients))
        return selected_clients

    
        

    def print_clients_info(self):
        print("Current number of clients: ", self.current_num_clients)
        for client in self.clients:
            client.print_client_info()
    
    def sample_data_with_replacement(self, num_clients):
        client_data_lists = []
        training_size = self.x_train.shape[0]
        for i in range(num_clients):
            chosen_indices = random.sample([idx for idx in range(training_size)], self.num_data_samples_per_client)
            client_X = self.x_train[chosen_indices]
            client_Y = self.y_train[chosen_indices]

            client_X_train, client_X_valid, client_Y_train, client_Y_valid = train_test_split(client_X, client_Y, 
                                                                                              test_size = self.option['client_valid_ratio'],
                                                                                              random_state=self.option['seed'])
            # print(client_X_train)
            client_train_dataset = XYDataset(client_X_train, client_Y_train)
            client_valid_dataset = XYDataset(client_X_valid, client_Y_valid)
            # print(client_X_train.shape, client_X_valid.shape, client_Y_train.shape, client_Y_valid.shape)

            client_data_lists.append( (client_train_dataset, client_valid_dataset) )
        
        return client_data_lists


    def sample_data_without_replacement(self, num_clients):
        client_data_lists = []
        training_size = self.x_train.shape[0]
        # print("X train", self.x_train.shape)
        if self.option['non_iid_classes'] == 0:
            client_indices_split = np.split(np.array([idx for idx in range(training_size)]), num_clients)
            for i in range(num_clients):
                chosen_indices = client_indices_split[i]
                client_X = self.x_train[chosen_indices]
                client_Y = self.y_train[chosen_indices]
                # print(client_X.shape,client_Y.shape)

                client_X_train, client_X_valid, client_Y_train, client_Y_valid = train_test_split(client_X, client_Y, 
                                                                                                test_size = self.option['client_valid_ratio'])
                # print("Client X train",client_X_train)

                client_train_dataset = XYDataset(client_X_train, client_Y_train)
                client_valid_dataset = XYDataset(client_X_valid, client_Y_valid)
                # print(client_X_train.shape, client_X_valid.shape, client_Y_train.shape, client_Y_valid.shape)

                client_data_lists.append( (client_train_dataset, client_valid_dataset) )
        
        elif self.option['non_iid_classes'] == 1:
            non_iid_data_lists = []
            all_classes = list(np.unique(self.y_train))
            num_classes = len(all_classes)
            # print(all_classes, num_classes)
            num_partitions_per_class = num_clients // num_classes
            partition_size = self.x_train.shape[0] // num_clients
            print("Number of partitions per class", num_partitions_per_class)
            print("partition size", partition_size)
            # print(self.y_train.shape)
            for label in all_classes:
                label_indices = np.argwhere(self.y_train == label)
                x_train_label = self.x_train[label_indices].squeeze(0)
                y_train_label = self.y_train[label_indices]
                y_train_label = y_train_label.squeeze(0)
                print(x_train_label.shape, y_train_label.shape)
                for i in range(num_partitions_per_class):
                    x_train_label_partition = x_train_label[partition_size * i: partition_size * (i+1)]
                    y_train_label_partition = y_train_label[partition_size * i: partition_size * (i+1)]
                    print("partition shape: ", x_train_label_partition.shape, y_train_label_partition.shape)
                    client_X_train, client_X_valid, client_Y_train, client_Y_valid = train_test_split(x_train_label_partition, y_train_label_partition, 
                                                                                                    test_size = self.option['client_valid_ratio'])

                    # print(client_X_train.shape, client_Y_train.shape, client_X_valid.shape)
                    client_train_dataset = XYDataset(client_X_train, client_Y_train)
                    client_valid_dataset = XYDataset(client_X_valid, client_Y_valid)
                    non_iid_data_lists.append( (client_train_dataset, client_valid_dataset) )
            client_data_lists = non_iid_data_lists

                    # print(y_train_label_partition, x_train_label_partition, x_train_label_partition.shape)
                # print(x_train_label[0:10], y_train_label[0:10]
            # print(np.unique(self.y_train))
    
        
        return client_data_lists
    
    
    def initialize_edges(self):
        cover_areas = [(self.left_road_limit + int(self.road_distance / self.num_edges) * i,
                         self.left_road_limit + int(self.road_distance / self.num_edges) * (i+1)) for i in range(self.num_edges)  ]
        name_lists = ['e' + str(client_id) for client_id in range(self.num_edges)]
        self.edges = []
        for i in range(self.num_edges):
            edge = EdgeServer(self.option, model = copy.deepcopy(self.model)
                              , cover_area=cover_areas[i], name=name_lists[i], test_data = None)
            self.edges.append(edge)


     
    def print_edges_info(self):
        print("Current number of edges: ", self.num_edges)
        for edge in self.edges:
            edge.print_edge_info()






class EdgeServer(BasicEdgeServer):
    def __init__(self, option, cover_area,  name='', train_data=None, valid_data=None):
        super(EdgeServer, self).__init__(option, cover_area,  name, train_data, valid_data)

    def print_edge_info(self):
        print('Edge {} - cover area: {}'.format(self.name,self.cover_area))

class MobileClient(BasicMobileClient):
    def __init__(self, option, location = 0,  velocity = 0, name='', train_data=None, valid_data=None):
        super(MobileClient, self).__init__(option, location,  name, train_data, valid_data)
        self.velocity = velocity
    
    def print_client_info(self):
        print('Client {} - current loc: {} - velocity: {} - training data size: {}'.format(self.name,self.location,self.velocity,
                                                                                           self.datavol))
    
    def update_location(self):
        self.location += self.velocity