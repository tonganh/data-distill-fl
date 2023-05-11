import random
from utils import fmodule
import sys
sys.path.append('..')
from  algorithm.fedbase import BasicServer, BasicClient
import copy
import math
import numpy as np


class BasicCloudServer(BasicServer):
    def __init__(self, option, model ,clients,test_data = None):
        super(BasicCloudServer, self).__init__(option, model, clients, test_data)
        # self.clients = []
        # print(clients)
        self.road_distance = option['road_distance'] 
        self.left_road_limit, self.right_road_limit =  - self.road_distance / 2, self.road_distance / 2
        
        self.mean_num_clients = option['num_clients']
        self.std_num_clients = option['std_num_clients']
        self.current_num_clients = self.mean_num_clients

        self.num_edges = option['num_edges']
        self.edges = []

        self.mean_velocity = option['mean_velocity']
        self.std_velocity = option['std_velocity']
        self.edge_update_frequency = option['edge_update_frequency']

        # self.sample_with_replacement = option['sample_with_replacement']
        self.client_edge_mapping = {}
        self.unused_clients_queue = []

        self.option = option

        # List to store clients currently in range
        self.selected_clients = []
        # List to store clients currently out of range
        self.unused_clients_queue = []
        # print("Clients" , self.clients)


    
    def global_update_location(self):
        new_client_list = []
        for client in self.clients:
            # client.print_client_info()
            # print(client.location)
            client.update_location()
            # print(client.location)
            new_client_list.append(client)
        self.clients = new_client_list

    def assign_client_to_server(self):
        client_buffer = self.clients.copy()
        for edge in self.edges:
            edge_area = edge.cover_area
            edge_name = edge.name
            if edge not in self.client_edge_mapping:
                self.client_edge_mapping[edge_name] = []
            for client in client_buffer:
                if edge_area[0] <= client.location <= edge_area[1]:
                    self.client_edge_mapping[edge_name].append(client.name)
        # print(self.client_edge_mapping)
                    # print(self.client_edge_mapping)
        
    def update_client_list(self):
        filtered_client_list = []
        filtered = 0
        for client in self.clients:
            # client.print_client_info()
            if self.left_road_limit <= client.location <= self.right_road_limit:
                filtered_client_list.append(client)
                # print(True)
            else:
                self.unused_clients_queue.append(client)
                filtered +=1
                # print(False)
        # print("Number of filtered clients",filtered)
        self.clients = filtered_client_list
        if len(self.clients) < self.mean_num_clients - self.std_num_clients:
            self.current_num_clients = np.random.randint(low = self.mean_num_clients - self.std_num_clients, high=self.mean_num_clients+self.std_num_clients + 1,
                                                        size=1)[0]
            num_clients_to_readd = self.current_num_clients - len(self.clients)
            if num_clients_to_readd < len(self.unused_clients_queue):
                clients_to_readd = random.sample(self.unused_clients_queue, k = num_clients_to_readd)
                for client in clients_to_readd:
                    client.location =np.random.randint( self.left_road_limit, self.right_road_limit, size =1)[0]
                    self.clients.append(client)
                    # client.location = client
            else:
                clients_to_readd = self.unused_clients_queue
                for client in clients_to_readd:
                    client.location =np.random.randint( self.left_road_limit, self.right_road_limit, size =1)[0]
                    self.clients.append(client)

    def get_current_num_clients(self):
        self.current_num_clients = np.random.randint()
    
    def initialize_clients_location_velocity(self):
        new_client_list = []
        locations = np.random.randint( self.left_road_limit, self.right_road_limit, size = len(self.clients))
        if self.option['mean_velocity'] != 0:
            velocities_absolute = np.random.randint( self.mean_velocity - self.std_velocity, self.mean_velocity + self.std_velocity, size = len(self.clients))
            velocities_direction = np.array([random.choice([-1,1]) for i in range(len(self.clients))])
            velocities = velocities_absolute * velocities_direction
            # print(velocities_direction, velocities)
        else:
            velocities = np.array([0 for i in range(len(self.clients))])

        for i  in range(len(self.clients)):
            client = self.clients[i]
            client.location = locations[i] 
            client.velocity = velocities[i]
            new_client_list.append(client)
        
        self.clients = new_client_list
        


    def print_edges_info(self):
        print("Current number of edges: ", self.num_edges)
        for edge in self.edges:
            edge.print_edge_info()
    
    # def print_client_info(self):
    #     for client in self.clients
            

    def initialize(self):
        # self.initialize_edges()
        self.assign_client_to_server()
        self.initialize_clients_location_velocity()


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
        self.total_datavol = 0
        # self.list_clients = []
    

    # def update_list_clients(self,clients):
    #     self.list_clients = clients

    

class BasicMobileClient(BasicClient):
    def __init__(self, option, location = 0, velocity = 0, name='', train_data=None, valid_data=None):
        super(BasicMobileClient, self).__init__(option, name, train_data, valid_data)
        self.location = location
        self.velocity = velocity
        # self.mean_velocity = mean_velocity
        # self.std_velocity = std_velocity
        # self.current_velocity = mean_velocity
    
    # def get_current_velocity(self):
    #     self.current_velocity = np.random.randint(low=self.mean_velocity - self.std_velocity, high=self.mean_velocity + self.std_velocity, size = 1)[0]

    def update_location(self):
        self.location += self.velocity

    def get_location(self):
        return self.location
    

