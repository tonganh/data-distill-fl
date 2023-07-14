import random

import torch
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
from .mobile_fl_utils import model_weight_divergence, kl_divergence, calculate_kl_div_from_data

class CloudServer(BasicCloudServer):
    def __init__(self, option, model ,clients,test_data = None):
        super(CloudServer, self).__init__( option, model,clients,test_data )
        self.initialize()    
        

    def communicate(self, edges):
        """
        The whole simulating communication procedure with the selected clients.
        This part supports for simulating the client dropping out.
        :param
            selected_clients: the clients to communicate with
        :return
            :the unpacked response from clients that is created ny self.unpack()
        """
        packages_received_from_edges = []
        if self.num_threads <= 1:
            # computing iteratively
            for edge in edges:
                response_from_edge = self.communicate_with(edge)
                packages_received_from_edges.append(response_from_edge)
    
        else:
            # computing in parallel
            pool = ThreadPool(min(self.num_threads, len(edges)))
            packages_received_from_edges = pool.map(self.communicate_with, edges)
            pool.close()
            pool.join()
        # count the clients not dropping
        # self.selected_clients = [selected_clients[i] for i in range(len(selected_clients)) if packages_received_from_clients[i]]
        # packages_received_from_edges = [pi for pi in packages_received_from_clients if pi]
        return self.unpack(packages_received_from_edges)

    def communicate_with(self, edge):
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
        reply = edge.reply(svr_pkg)
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

    # def sample(self):
    #     """Sample the clients.
    #     :param
    #         replacement: sample with replacement or not
    #     :return
    #         a list of the ids of the selected clients
    #     """
    #     # print("Sampling selected clients")
    #     all_clients = [cid for cid in range(self.num_clients)]

    #     # print("Done all clients")
    #     selected_clients = []
    #     # collect all the active clients at this round and wait for at least one client is active and
    #     active_clients = []
    #     active_clients = self.clients
    #     self.clients_per_round =  max(int(self.num_clients * self.option['proportion']), 1)

    #     # while(len(active_clients)<1):
    #     #     active_clients = [cid for cid in range(self.num_clients) if self.clients[cid].is_active()]
    #     # print("DOne collect all the active clients")
    #     # sample clients
    #     if self.sample_option == 'active':
    #         # select all the active clients without sampling
    #         selected_clients = active_clients
    #     if self.sample_option == 'uniform':
    #         # original sample proposed by fedavg
    #         # print(self.clients_per_round)
    #         # print(len(self.clients))
    #         selected_clients = list(np.random.choice(active_clients,self.clients_per_round, replace=False))

    #     elif self.sample_option =='md':
    #         # the default setting that is introduced by FedProx
    #         selected_clients = list(np.random.choice(all_clients, self.clients_per_round, replace=True, p=[nk / self.data_vol for nk in self.client_vols]))
    #     # drop the selected but inactive clients
    #     selected_clients = list(set(active_clients).intersection(selected_clients))
    #     return selected_clients



    # def create_clients(self, num_clients):
    #     locations = np.random.randint( self.left_road_limit, self.right_road_limit, size = num_clients)
    #     if self.option['mean_velocity'] != 0:
    #         velocities = np.random.randint( - (self.mean_velocity + self.std_velocity), self.mean_velocity + self.std_velocity, size = num_clients)
    #     else:
    #         velocities = np.array([0 for i in range(num_clients)])
    #     name_lists = ['c' + str(client_id) for client_id in range(num_clients)]
    #     if self.option['sample_with_replacement']:
    #         client_data_lists = self.sample_data_with_replacement(num_clients=num_clients)
    #     else:
    #         client_data_lists = self.sample_data_without_replacement(num_clients=num_clients)

    #     new_clients = []
    #     for i in range(num_clients):
    #         client_train_data, client_valid_data = client_data_lists[i]
    #         client = MobileClient(self.option, location=locations[i], velocity=velocities[i], name=name_lists[i], 
    #                               train_data=client_train_data, valid_data = client_valid_data)
    #         new_clients.append(client)
        
    #     return new_clients
    
    # def initialize_clients(self):
    #     self.clients = self.create_clients(self.current_num_clients)
    
        
    def initialize_edges(self):
        cover_areas = [(self.left_road_limit + int(self.road_distance / self.num_edges) * i,
                         self.left_road_limit + int(self.road_distance / self.num_edges) * (i+1)) for i in range(self.num_edges)  ]
        name_lists = ['e' + str(client_id) for client_id in range(self.num_edges)]
        self.edges = []
        for i in range(self.num_edges):
            edge = EdgeServer(self.option, model = copy.deepcopy(self.model)
                              , cover_area=cover_areas[i], name=name_lists[i], test_data = None)
            self.edges.append(edge)

    def print_clients_info(self):
        print("Current number of clients: ", self.current_num_clients)
        for client in self.clients:
            client.print_client_info()

    def initialize(self):
        self.initialize_edges()
        self.assign_client_to_server()
        self.initialize_clients_location_velocity()


    # def sample_data_with_replacement(self, num_clients):
    #     client_data_lists = []
    #     training_size = self.x_train.shape[0]
    #     for i in range(num_clients):
    #         chosen_indices = random.sample([idx for idx in range(training_size)], self.num_data_samples_per_client)
    #         client_X = self.x_train[chosen_indices]
    #         client_Y = self.y_train[chosen_indices]

    #         client_X_train, client_X_valid, client_Y_train, client_Y_valid = train_test_split(client_X, client_Y, 
    #                                                                                           test_size = self.option['client_valid_ratio'],
    #                                                                                           random_state=self.option['seed'])
    #         # print(client_X_train)
    #         client_train_dataset = XYDataset(client_X_train, client_Y_train)
    #         client_valid_dataset = XYDataset(client_X_valid, client_Y_valid)
    #         # print(client_X_train.shape, client_X_valid.shape, client_Y_train.shape, client_Y_valid.shape)

    #         client_data_lists.append( (client_train_dataset, client_valid_dataset) )
        
    #     return client_data_lists


    # def sample_data_without_replacement(self, num_clients):
    #     client_data_lists = []
    #     training_size = self.x_train.shape[0]
    #     # print("X train", self.x_train.shape)
    #     if self.option['non_iid_classes'] == 0:
    #         client_indices_split = np.split(np.array([idx for idx in range(training_size)]), num_clients)
    #         for i in range(num_clients):
    #             chosen_indices = client_indices_split[i]
    #             client_X = self.x_train[chosen_indices]
    #             client_Y = self.y_train[chosen_indices]
    #             # print(client_X.shape,client_Y.shape)

    #             client_X_train, client_X_valid, client_Y_train, client_Y_valid = train_test_split(client_X, client_Y, 
    #                                                                                             test_size = self.option['client_valid_ratio'])
    #             # print("Client X train",client_X_train)

    #             client_train_dataset = XYDataset(client_X_train, client_Y_train)
    #             client_valid_dataset = XYDataset(client_X_valid, client_Y_valid)
    #             # print(client_X_train.shape, client_X_valid.shape, client_Y_train.shape, client_Y_valid.shape)

    #             client_data_lists.append( (client_train_dataset, client_valid_dataset) )
        
    #     else:
    #         non_iid_data_lists = []
    #         all_classes = list(np.unique(self.y_train))
    #         num_classes = len(all_classes)
    #         # print(all_classes, num_classes)
    #         num_partitions_per_class = num_clients // num_classes
    #         partition_size = self.x_train.shape[0] // num_clients
    #         print("Number of partitions per class", num_partitions_per_class)
    #         print("partition size", partition_size)
    #         # print(self.y_train.shape)
    #         for label in all_classes:
    #             label_indices = np.argwhere(self.y_train == label)
    #             x_train_label = self.x_train[label_indices].squeeze(0)
    #             y_train_label = self.y_train[label_indices]
    #             y_train_label = y_train_label.squeeze(0)
    #             print(x_train_label.shape, y_train_label.shape)
    #             for i in range(num_partitions_per_class):
    #                 x_train_label_partition = x_train_label[partition_size * i: partition_size * (i+1)]
    #                 y_train_label_partition = y_train_label[partition_size * i: partition_size * (i+1)]
    #                 print("partition shape: ", x_train_label_partition.shape, y_train_label_partition.shape)
    #                 client_X_train, client_X_valid, client_Y_train, client_Y_valid = train_test_split(x_train_label_partition, y_train_label_partition, 
    #                                                                                                 test_size = self.option['client_valid_ratio'])

    #                 # print(client_X_train.shape, client_Y_train.shape, client_X_valid.shape)
    #                 client_train_dataset = XYDataset(client_X_train, client_Y_train)
    #                 client_valid_dataset = XYDataset(client_X_valid, client_Y_valid)
    #                 non_iid_data_lists.append( (client_train_dataset, client_valid_dataset) )
    #         client_data_lists = non_iid_data_lists

    #                 # print(y_train_label_partition, x_train_label_partition, x_train_label_partition.shape)
    #             # print(x_train_label[0:10], y_train_label[0:10]
    #         # print(np.unique(self.y_train))

        
    #     return client_data_lists
        
        # print(self.client_edge_mapping)




class EdgeServer(BasicEdgeServer):
    def __init__(self, option,model,cover_area, name = '', clients = [], test_data=None):
        super(EdgeServer, self).__init__(option,model,cover_area, name , clients , test_data)
        self.clients = []
        self.agg_option = self.option['aggregate']

    def update_client_list(self,clients):
        self.clients = clients
    
    def get_data(self):
        all_edge_data = []
        for client in self.clients:
            # print(client.train_data.X.shape)
            # return client.train_data.X
            all_edge_data.append(client.train_data.X)
        
        edge_data = torch.cat(all_edge_data,0)
        # print(edge_data.shape)
        return edge_data
    
    # def get_client_distribution()

    def print_edge_info(self):
        print('Edge {} - cover area: {}'.format(self.name,self.cover_area))

    def communicate(self, clients):
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
            for client in clients:
                response_from_edge = self.communicate_with(client)
                packages_received_from_clients.append(response_from_edge)
    
        else:
            # computing in parallel
            pool = ThreadPool(min(self.num_threads, len(clients)))
            packages_received_from_clients = pool.map(self.communicate_with,clients)
            pool.close()
            pool.join()
        # count the clients not dropping
        # self.selected_clients = [selected_clients[i] for i in range(len(selected_clients)) if packages_received_from_clients[i]]
        # packages_received_from_edges = [pi for pi in packages_received_from_clients if pi]
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
        edge_pkg = self.pack()
        # listen for the client's response and return None if the client drops out
        # if self.clients[client_id].is_drop(): return None
        reply = client.reply(edge_pkg)
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

    def unpack_svr(self, received_pkg):
        """
        Unpack the package received from the cloud server
        :param
            received_pkg: a dict contains the global model as default
        :return:
            the unpacked information that can be rewritten
        """
        # unpack the received package
        return received_pkg['model']

    # def reply(self, svr_pkg):
    #     """
    #     Reply to server with the transmitted package.
    #     The whole local procedure should be planned here.
    #     The standard form consists of three procedure:
    #     unpacking the server_package to obtain the global model,
    #     training the global model, and finally packing the improved
    #     model into client_package.
    #     :param
    #         svr_pkg: the package received from the server
    #     :return:
    #         client_pkg: the package to be send to the server
    #     """
    #     # print("In reply function of client")
    #     # print(svr_pkg)
    #     model = self.unpack_svr(svr_pkg)
    #     # print("CLient unpacked to package")
    #     # loss = self.train_loss(model)
    #     loss = 0
    #     print("Client evaluated the train losss")
    #     self.train(model)
    #     print("Client trained the model")
    #     cpkg = self.pack(model, loss)
    #     # print("Client packed and finished")
    #     return cpkg


class MobileClient(BasicMobileClient):
    def __init__(self, option, location = 0,  velocity = 0, name='', train_data=None, valid_data=None):
        super(MobileClient, self).__init__(option, location, velocity,  name, train_data, valid_data)
        # self.velocity = velocity
        self.associated_server = None
    
    def print_client_info(self):
        print('Client {} - current loc: {} - velocity: {} - training data size: {}'.format(self.name,self.location,self.velocity,
                                                                                           self.datavol))
    
