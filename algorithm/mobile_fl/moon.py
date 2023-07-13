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
import torch.nn as nn

class CloudServer(BasicCloudServer):
    def __init__(self, option, model ,clients,test_data = None):
        super(CloudServer, self).__init__( option, model,clients,test_data )
        self.initialize()


    def run(self):
        """
        Start the federated learning symtem where the global model is trained iteratively.
        """
        logger.time_start('Total Time Cost')
        for round in range(self.num_rounds+1):
            print("--------------Round {}--------------".format(round))
            logger.time_start('Time Cost')
            # print(self.clients)
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
        # First, distill all data on clients' side
        # for client in self.clients:
        #     client.distill_data()

        num_iterations_start_remove = 50
        num_clients_removed = 2
        after_num_itr_remove = 5
        if t >= num_iterations_start_remove and t%after_num_itr_remove==0:
            self.delete_clients(num_clients_removed)

        # sample clients: MD sampling as default but with replacement=False
        # print("Iterating")
        self.global_update_location()
        # print("Done updating location")
        self.update_client_list()
        # print("Done updating client_list")
        self.assign_client_to_server()
        # print("Done assigning client to sercer")

        # self.selected_clients = self.sample()
        self.selected_clients = self.clients
        # print("Selected clients", [client.name for client in self.selected_clients])

        # first, aggregate the edges with their clientss
        # for client in self.selected_clients:
        #     client.print_client_info()
        all_client_train_losses = []
        all_client_valid_losses = []
        all_client_train_metrics = []
        all_client_valid_metrics = []

        for edge in self.edges:

            # print(f"Edge: {edge.name} - clients {self.client_edge_mapping[edge.name]}" )
            clients_chosen_in_edge =     list(np.random.choice(self.client_edge_mapping[edge.name],
                                                               int(len(self.client_edge_mapping[edge.name]) * self.option['proportion']), replace=False))

            # print(f"Edge: {edge.name} - clients {clients_chosen_in_edge}" )


            aggregated_clients = []
            for client in self.selected_clients:
                if client.name in clients_chosen_in_edge:
                    aggregated_clients.append(client)
            if len(aggregated_clients) > 0:
                # print(aggregated_clients)
                # print(edge.communicate(aggregated_clients))
                aggregated_clients_models , (agg_clients_train_losses, 
                                             agg_clients_valid_losses, 
                                             agg_clients_train_accs, 
                                             agg_clients_valid_accs)= edge.communicate(aggregated_clients)
                
                edge_total_datavol = sum([client.datavol for client in aggregated_clients])
                edge.total_datavol = edge_total_datavol
                aggregation_weights = [client.datavol / edge_total_datavol for client in aggregated_clients]
                # print(len(aggregation_weights), len(aggregated_clients_models))
                edge.model =  self.aggregate(aggregated_clients_models, p = aggregation_weights)

                all_client_train_losses.extend(agg_clients_train_losses)
                all_client_valid_losses.extend(agg_clients_valid_losses)
                all_client_train_metrics.extend(agg_clients_train_accs)
                all_client_valid_metrics.extend(agg_clients_valid_accs)
        
        self.client_train_losses.append(sum(all_client_train_losses) / len(all_client_train_losses))
        self.client_valid_losses.append(sum(all_client_valid_losses) / len(all_client_valid_losses))
        self.client_train_metrics.append(sum(all_client_train_metrics) / len(all_client_train_metrics))
        self.client_valid_metrics.append(sum(all_client_valid_metrics) / len(all_client_valid_metrics))

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
            sum_datavol = sum([edge.total_datavol for edge in self.edges])
            edge_weights = [edge.total_datavol / sum_datavol for edge in self.edges]
            self.model = self.aggregate(models, p = edge_weights)

            for edge in self.edges:
                edge.model = copy.deepcopy(self.model)

        edges_models_list = []
        for edge in self.edges:
                edges_models_list.append(copy.deepcopy(edge.model))
     





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

class ModelContrastiveLoss(nn.Module):
    def __init__(self, mu=0.01, tau=3):
        super(ModelContrastiveLoss, self).__init__()
        self.mu = mu
        self.tau = tau
        self.ce = nn.CrossEntropyLoss()
        self.sim = nn.CosineSimilarity(dim=-1)

    def forward(self, logits, targets, z, z_prev, z_g):
        device = logits.device
        loss1 = self.ce(logits, targets)

        positive = self.sim(z, z_g).reshape(-1, 1)
        negative = self.sim(z, z_prev).reshape(-1, 1)
        moon_logits = torch.cat([positive, negative], dim=1)
        moon_logits /= self.tau
        moon_labels = torch.zeros(z.size(0)).to(device).long()

        loss2 = self.ce(moon_logits, moon_labels)

        total_loss = loss1 + self.mu * loss2

        return total_loss

class MobileClient(BasicMobileClient):
    def __init__(self, option, location = 0,  velocity = 0, name='', train_data=None, valid_data=None):
        super(MobileClient, self).__init__(option, location, velocity,  name, train_data, valid_data)
        # self.velocity = velocity
        self.associated_server = None
        self.prev_model = None
        self.moon_criterion = ModelContrastiveLoss()

    def train(self,model):
        """Local training"""

        # Keep global model and prev local model
        # self._keep_global()
        # self._keep_prev_local()
        if self.prev_model == None:
            self.prev_model = copy.deepcopy(model)
        src_model = copy.deepcopy(model)
        src_model.freeze_grad()
        self.prev_model.freeze_grad()

        model.train()
        # model.to(self.device)
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)

        for _ in range(self.epochs):
            for batch_idx, batch_data in enumerate(data_loader):
                model.zero_grad()

                tdata = self.calculator.data_to_device(batch_data)
                input, target = tdata[0], tdata[1].type(torch.LongTensor)


                target = target.to(input.device)

                output,z = model.pred_and_rep(input)
                _, z_prev = self.prev_model.pred_and_rep(input)
                _, z_g = src_model.pred_and_rep(input)


                loss = self.moon_criterion(output, target, z, z_prev, z_g)

                # backward pass
                loss.backward()
                optimizer.step()


        return 
    def print_client_info(self):
        print('Client {} - current loc: {} - velocity: {} - training data size: {}'.format(self.name,self.location,self.velocity,
                                                                                           self.datavol))
    
