import random

import torch
from utils import fmodule
import sys
sys.path.append('..')
from .fedbase_mobile_distill  import BasicCloudServer, BasicEdge, BasicMobileClient
from benchmark.toolkits import XYDataset
import copy
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from main_distill import logger
import os
from tqdm import tqdm
from multiprocessing import Pool as ThreadPool
from algorithm.distill_fl.distill_utils.distiller import Distiller


class CloudServer(BasicCloudServer):
    def __init__(self, option, model ,clients,test_data = None):
        super(CloudServer, self).__init__( option, model,clients,test_data )
        self.initialize()

        
        self.avg_edge_train_losses = []
        self.avg_edge_valid_losses = []
        self.avg_edge_train_metrics = []
        self.avg_edge_valid_metrics = []
        self.edge_metrics = {}

    def iterate(self, t):
        """
        The standard iteration of each federated round that contains three
        necessary procedure in FL: client selection, communication and model aggregation.
        :param
            t: the number of current round
        """
        # First, distill all data on clients' side
        # for client in self.clients:
        #     client.distill_data()


        # sample clients: MD sampling as default but with replacement=False
        # print("Iterating")
        self.global_update_location()
        # print("Done updating location")
        self.update_client_list()
        # print("Done updating client_list")
        self.assign_client_to_server()
        # print("Done assigning client to sercer")

        self.selected_clients = self.sample()
        print("Selected clients", len(self.selected_clients))
        print("Transfer data of client selected", len(self.selected_clients))

        # first, aggregate the edges with their clientss
        # for client in self.selected_clients:
        #     client.print_client_info()
        for edge in self.edges:
            aggregated_clients = []
            for client in self.selected_clients:
                if client.name in self.client_edge_mapping[edge.name]:
                    aggregated_clients.append(client)
            if len(aggregated_clients) > 0:
                edge.collect_distilled_data_from_client(aggregated_clients)
        
        
        models, (edge_names, train_losses, valid_losses, train_acc, valid_acc) = self.communicate(self.edges)
        
        all_edge_train_losses = []
        all_edge_valid_losses = []
        all_edge_train_metrics = []
        all_edge_valid_metrics = []

        for i in range(len(edge_names)):
            edge_name = edge_names[i]
            edge_train_loss = train_losses[i]
            edge_valid_loss = valid_losses[i]
            edge_train_acc = train_acc[i]
            edge_valid_acc = valid_acc[i]
            if edge_name in self.edge_metrics.keys():
                self.edge_metrics[edge_name].append([t,edge_train_loss, edge_train_acc, edge_valid_loss, edge_valid_acc])
            else:
                self.edge_metrics[edge_name] = [['Round','train_losses', 'train_accs', 'val_Losses', 'val_accs']]

            all_edge_train_losses.append(edge_train_loss)
            all_edge_valid_losses.append(edge_valid_loss)
            all_edge_train_metrics.append(edge_train_acc)
            all_edge_valid_metrics.append(edge_valid_acc)
        
        self.avg_edge_train_losses.append(sum(all_edge_train_losses) / len(all_edge_train_losses))
        self.avg_edge_valid_losses.append(sum(all_edge_valid_losses) / len(all_edge_valid_losses))
        self.avg_edge_train_metrics.append(sum(all_edge_train_metrics) / len(all_edge_train_metrics))
        self.avg_edge_valid_metrics.append(sum(all_edge_valid_metrics) / len(all_edge_valid_metrics))


        
            # else:
            #     print('No aggregated clients')
        # models, train_losses = self.communicate(self.edges)

        # print("Done a training step")
        # check whether all the clients have dropped out, because the dropped clients will be deleted from self.selected_clients
        if not self.selected_clients: return
        # aggregate: pk = 1/K as default where K=len(selected_clients)
        # models = [edge.model for edge in self.edges]
        if t % self.edge_update_frequency == 0:
            models = [edge.model for edge in self.edges]
            sum_datavol = sum([edge.datavol for edge in self.edges])
            edge_weights = [edge.datavol / sum_datavol for edge in self.edges]
            self.model = self.aggregate(models, p = edge_weights)

            for edge in self.edges:
                edge.model = copy.deepcopy(self.model)


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




class EdgeServer(BasicEdge):
    def __init__(self, option,model,cover_area, name = '', clients = [], test_data=None):
        super(EdgeServer, self).__init__(option,model,cover_area, name , clients , test_data)
        self.clients = []

    def update_client_list(self,clients):
        import pdb; pdb.set_trace()
        self.clients = clients
    
    def print_edge_info(self):
        print('Edge {} - cover area: {}'.format(self.name,self.cover_area))


class MobileClient(BasicMobileClient):
    def __init__(self, option, location = 0,  velocity = 0, name='', train_data=None, valid_data=None):
        super(MobileClient, self).__init__(option, location, velocity,  name, train_data, valid_data)
        # self.velocity = velocity
        self.associated_server = None
    
    def print_client_info(self):
        print('Client {} - current loc: {} - velocity: {} - training data size: {}'.format(self.name,self.location,self.velocity,
                                                                                           self.datavol))
    
