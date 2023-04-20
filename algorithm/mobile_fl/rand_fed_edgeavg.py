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
        # self.print_clients_info()
        # self.print_edges_info()
        self.unused_clients_queue = []
        print("Done with initialization")


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


        # first, aggregate the edges with their clients
        for edge in self.edges:
            aggregated_clients = []
            for client in self.selected_clients:
                if client.name in self.client_edge_mapping[edge.name]:
                    aggregated_clients.append(client)
            if len(aggregated_clients) > 0:
                aggregated_clients_models , _= edge.communicate(aggregated_clients)
                edge.model =  self.aggregate(aggregated_clients_models, p = [1.0  for cid in range(len(aggregated_clients))])
        # models, train_losses = self.communicate(self.edges)

        # print("Done a training step")
        # check whether all the clients have dropped out, because the dropped clients will be deleted from self.selected_clients
        if not self.selected_clients: return
        # aggregate: pk = 1/K as default where K=len(selected_clients)
        # models = [edge.model for edge in self.edges]
        if t % self.edge_update_frequency == 0:
            models = [edge.model for edge in self.edges]
            self.model = self.aggregate(models, p = [1.0  for cid in range(len(self.edges))])



    

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



    def create_clients(self, num_clients):
        locations = np.random.randint( self.left_road_limit, self.right_road_limit, size = num_clients)
        if self.option['mean_velocity'] != 0:
            velocities = np.random.randint( - (self.mean_velocity + self.std_velocity), self.mean_velocity + self.std_velocity, size = num_clients)
        else:
            velocities = np.array([0 for i in range(num_clients)])
        name_lists = ['c' + str(client_id) for client_id in range(num_clients)]
        if self.option['sample_with_replacement']:
            client_data_lists = self.sample_data_with_replacement(num_clients=num_clients)
        else:
            client_data_lists = self.sample_data_without_replacement(num_clients=num_clients)

        new_clients = []
        for i in range(num_clients):
            client_train_data, client_valid_data = client_data_lists[i]
            client = MobileClient(self.option, location=locations[i], velocity=velocities[i], name=name_lists[i], 
                                  train_data=client_train_data, valid_data = client_valid_data)
            new_clients.append(client)
        
        return new_clients
    
    def initialize_clients(self):
        self.clients = self.create_clients(self.current_num_clients)
    
        

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
        client_indices_split = np.split(np.array([idx for idx in range(training_size)]), num_clients)
        for i in range(num_clients):
            chosen_indices = client_indices_split[i]
            client_X = self.x_train[chosen_indices]
            client_Y = self.y_train[chosen_indices]

            client_X_train, client_X_valid, client_Y_train, client_Y_valid = train_test_split(client_X, client_Y, 
                                                                                              test_size = self.option['client_valid_ratio'])
            # print("Client X train",client_X_train)

            client_train_dataset = XYDataset(client_X_train, client_Y_train)
            client_valid_dataset = XYDataset(client_X_valid, client_Y_valid)
            # print(client_X_train.shape, client_X_valid.shape, client_Y_train.shape, client_Y_valid.shape)

            client_data_lists.append( (client_train_dataset, client_valid_dataset) )

        
        return client_data_lists
    

    def global_update_location(self):
        new_client_list = []
        for client in self.clients:
            client.update_location()
            new_client_list.append(client)
        self.clients = new_client_list

    
    def update_client_list(self):
        filtered_client_list = []
        filtered = 0
        for client in self.clients:
            if self.left_road_limit <= client.location <= self.right_road_limit:
                filtered_client_list.append(client)
            else:
                self.unused_clients_queue.append(client)
                filtered +=1
        # print("Number of filtered clients",filtered)
        self.clients = filtered_client_list
        if len(self.clients) < self.mean_num_clients - self.std_num_clients:
            self.current_num_clients = np.random.randint(low = self.mean_num_clients - self.std_num_clients, high=self.mean_num_clients+self.std_num_clients + 1,
                                                        size=1)[0]
            num_clients_to_readd = self.current_num_clients - len(self.clients)
            if num_clients_to_readd < len(self.unused_clients_queue):
                clients_to_readd = random.sample(self.unused_clients_queue, k = num_clients_to_readd)
                self.clients.extend(clients_to_readd)
            else:
                self.clients.extend(self.unused_clients_queue)

        
        self.unused_clients_queue = list(set(self.unused_clients_queue) - set(self.clients))
        print("Number of unused clients", len(self.unused_clients_queue))



            # self.sample_new_clients(num_new_clients=self.current_num_clients - len(filtered_client_list))
     



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



    def initialize(self):
        self.initialize_clients()
        self.initialize_edges()
        self.assign_client_to_server()


class EdgeServer(BasicEdgeServer):
    def __init__(self, option,model,cover_area, name = '', clients = [], test_data=None):
        super(EdgeServer, self).__init__(option,model,cover_area, name , clients , test_data)

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
        svr_pkg = self.pack()
        # print(svr_pkg)
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

    def unpack_svr(self, received_pkg):
        """
        Unpack the package received from the server
        :param
            received_pkg: a dict contains the global model as default
        :return:
            the unpacked information that can be rewritten
        """
        # unpack the received package
        return received_pkg['model']

    def reply(self, svr_pkg):
        """
        Reply to server with the transmitted package.
        The whole local procedure should be planned here.
        The standard form consists of three procedure:
        unpacking the server_package to obtain the global model,
        training the global model, and finally packing the improved
        model into client_package.
        :param
            svr_pkg: the package received from the server
        :return:
            client_pkg: the package to be send to the server
        """
        # print("In reply function of client")
        # print(svr_pkg)
        model = self.unpack_svr(svr_pkg)
        # print("CLient unpacked to package")
        # loss = self.train_loss(model)
        loss = 0
        # print("Client evaluated the train losss")
        self.train(model)
        # print("Client trained the model")
        cpkg = self.pack(model, loss)
        # print("Client packed and finished")
        return cpkg


class MobileClient(BasicMobileClient):
    def __init__(self, option, location,  velocity, name='', train_data=None, valid_data=None):
        super(MobileClient, self).__init__(option, location,  name, train_data, valid_data)
        self.velocity = velocity
        self.option = option
    
    def print_client_info(self):
        print('Client {} - current loc: {} - velocity: {} - training data size: {}'.format(self.name,self.location,self.velocity,
                                                                                           self.datavol))
    
    def update_location(self):
        # self.location += self.velocity
        self.location  = np.random.randint(low=-self.option['road_distance']//2, high = self.option['road_distance']//2, size = 1)[0]