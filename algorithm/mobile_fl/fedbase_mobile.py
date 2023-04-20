from utils import fmodule
import sys
sys.path.append('..')
from  algorithm.fedbase import BasicServer, BasicClient
import copy
import math
import numpy as np


class BasicCloudServer(BasicServer):
    def __init__(self, option, model ,train_and_valid_data,test_data = None, clients = []):
        super(BasicCloudServer, self).__init__(option, model, clients, test_data)
        self.clients = []
        self.road_distance = option['road_distance']
        
        
        self.left_road_limit, self.right_road_limit =  - self.road_distance / 2, self.road_distance / 2
        
        
        self.mean_num_clients = option['num_clients']
        self.std_num_clients = option['std_num_clients']
        self.current_num_clients = self.mean_num_clients

        self.num_edges = option['num_edges']
        self.edges = []

        self.train_and_valid_data = train_and_valid_data
        self.clients = []



        self.mean_velocity = option['mean_velocity']
        self.std_velocity = option['std_velocity']

        self.train_and_valid_data = train_and_valid_data
        self.test_data = test_data

        self.edge_update_frequency = option['edge_update_frequency']

        self.sample_with_replacement = option['sample_with_replacement']
        self.client_edge_mapping = {}
        
        self.num_data_samples_per_client = len(self.train_and_valid_data) // self.mean_num_clients

        self.option = option

        self.x_train , self.y_train = self.train_and_valid_data.get_data()
        self.x_test, self.y_test = self.test_data.get_data()

        print(self.x_train.shape, self.y_train.shape, self.x_test.shape, self.y_test.shape)

        # self.intialize()

    def initialize_clients(self):
        pass


    def initialize_server_lists(self):
        return None
    

    def assign_client_to_server(self):
        pass



    def iterate(self, t):
        """
        The standard iteration of each federated round that contains three
        necessary procedure in FL: client selection, communication and model aggregation.
        :param
            t: the number of current round
        """
        # sample clients: MD sampling as default but with replacement=False
        # print("Iterating")
        self.selected_clients = self.sample()
        # print("Done sampling")
        # training
        models, train_losses = self.communicate(self.selected_clients)
        # print("Done a training step")
        # check whether all the clients have dropped out, because the dropped clients will be deleted from self.selected_clients
        if not self.selected_clients: return
        # aggregate: pk = 1/K as default where K=len(selected_clients)
        self.model = self.aggregate(models, p = [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients])
        return
    
    def global_update_location(self):
        # """Update the location of all clients"""
        # new_client_list = []
        # for client in self.clients:
        #     client.update_location()
        #     new_client_list.append(client)
        # self.clients = new_client_list
        pass
    
    def get_current_num_clients(self):
        self.current_num_clients = np.random.randint()

    def update_client_lists(self):
        """Filter out the clients that are too far away"""
        # self.global_update_location()
        # filtered_client_list = []
        # for client in self.clients:
        #     if self.left_road_limit <= client.get_location <= self.right_road_limit:
        #         filtered_client_list.append(client)
        
        # self.clients = filtered_client_list
        # self.current_num_clients = np.random.randint(low = self.mean_num_clients - self.std_num_clients, high=self.mean_num_clients+self.std_num_clients,
        #                                              size=1)[0]
        # if self.current_num_clients > len(filtered_client_list):
        #     self.sample_new_clients(num_new_clients=self.current_num_clients - len(filtered_client_list))
        pass
    
    def sample_new_clients(self, num_new_clients):
        # for i in range(num_new_clients):
        #     initial_location = self.
        pass

    def sample(self):
        pass        


class BasicEdgeServer(BasicServer):
    def __init__(self, option,model,cover_area, name = '', clients = [], test_data=None):
        super(BasicEdgeServer, self).__init__(option, model, clients, test_data)
        self.cover_area = cover_area
        self.name = name
        self.option = option
        # self.list_clients = []
    

    # def update_list_clients(self,clients):
    #     self.list_clients = clients

    

class BasicMobileClient(BasicClient):
    def __init__(self, option, location,  name='', train_data=None, valid_data=None):
        super(BasicMobileClient, self).__init__(option, name, train_data, valid_data)
        self.location = location
        # self.mean_velocity = mean_velocity
        # self.std_velocity = std_velocity
        # self.current_velocity = mean_velocity
    
    # def get_current_velocity(self):
    #     self.current_velocity = np.random.randint(low=self.mean_velocity - self.std_velocity, high=self.mean_velocity + self.std_velocity, size = 1)[0]

    def update_location(self):
        pass
    
    def get_location(self):
        return self.location
    

