from utils import fmodule
import sys
sys.path.append('..')
from  algorithm.fedbase import BasicServer, BasicClient
import copy
import math
import numpy as np


class BasicCloudServer(BasicServer):
    def __init__(self, option, model, mean_num_clients, std_num_clients, mean_velocity, std_velocity
                 ,train_and_valid_data, clients = [],
                 data_split_method = 'non_replacement',
                 edge_update_frequency = 1,
                 road_distance = 10000, test_data = None):
        super(BasicCloudServer, self).__init__(option, model, clients, test_data)
        self.clients = []
        self.road_distance = road_distance
        
        
        self.left_road_limit, self.right_road_limit =  - self.road_distance / 2, self.road_distance / 2
        
        
        self.mean_num_clients = mean_num_clients
        self.std_num_clients = std_num_clients
        self.current_num_clients = self.mean_num_clients


        self.train_and_valid_data = train_and_valid_data
        self.clients = []


        self.mean_velocity = mean_velocity
        self.std_velocity = std_velocity

        self.train_and_valid_data = train_and_valid_data
        self.test_data = test_data

        self.edge_update_frequency = edge_update_frequency

        self.client_edge_mapping = {}
        self.num_data_samples_per_client = len(self.train_and_valid_data) // self.mean_num_clients

    
    def initialize_clients(self):
        self.clients = []
        self.num_clients = len(self.clients)
        self.client_vols = [c.datavol for c in self.clients]
        self.data_vol = sum(self.client_vols)
        self.clients_buffer = [{} for _ in range(self.num_clients)]
        self.selected_clients = []
        return None


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
        """Update the location of all clients"""
        new_client_list = []
        for client in self.clients:
            client.update_location()
            new_client_list.append(client)
        self.clients = new_client_list
    
    def get_current_num_clients(self):
        self.current_num_clients = np.random.randint()
    def update_client_lists(self):
        """Filter out the clients that are too far away"""
        self.global_update_location()
        filtered_client_list = []
        for client in self.clients:
            if self.left_road_limit <= client.get_location <= self.right_road_limit:
                filtered_client_list.append(client)
        
        self.clients = filtered_client_list
        self.current_num_clients = np.random.randint(low = self.mean_num_clients - self.std_num_clients, high=self.mean_num_clients+self.std_num_clients,
                                                     size=1)[0]
        if self.current_num_clients > len(filtered_client_list):
            self.sample_new_clients(num_new_clients=self.current_num_clients - len(filtered_client_list))
    
    def sample_new_clients(self, num_new_clients):
        # for i in range(num_new_clients):
        #     initial_location = self.
        pass

    def sample(self):
        pass        


class BasicEdgeServer(BasicClient):
    def __init__(self, option,location, name='', train_data=None, valid_data=None):
        super(BasicEdgeServer, self).__init__(option, name, train_data, valid_data)
        self.location = location
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
        self.get_current_velocity()
        self.location = self.location + self.get_current_velocity()
    
    def get_location(self):
        return self.location
    

