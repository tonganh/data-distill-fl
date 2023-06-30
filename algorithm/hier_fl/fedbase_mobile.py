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
        # self.road_distance = option['road_distance'] 
        self.left_road_limit, self.right_road_limit =  - self.road_distance / 2, self.road_distance / 2
        
        self.mean_num_clients = option['num_clients']
        self.std_num_clients = option['std_num_clients']
        self.current_num_clients = self.mean_num_clients

        self.num_edges = option['num_edges']
        self.edges = []

        # self.mean_velocity = option['mean_velocity']
        # self.std_velocity = option['std_velocity']
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

        self.client_train_losses = []
        self.client_valid_losses = []
        self.client_train_metrics = []
        self.client_valid_metrics = []

    
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

        # print("Mean num clients: ", self.mean_num_clients)
        # print("std num clients: ", self.std_num_clients)
        # print("Number of clients from previous round" ,len(self.clients))

        # print("unused client queue: ", [client.name for client in self.unused_clients_queue])
        # print("Current clients: ", [client.name for client in self.clients])

        if len(self.clients) < self.mean_num_clients - self.std_num_clients:
            self.current_num_clients = np.random.randint(low = self.mean_num_clients - self.std_num_clients, high=self.mean_num_clients+self.std_num_clients + 1,
                                                        size=1)[0]

            # print("Chosen number of clients for this round" ,self.current_num_clients)

            num_clients_to_readd = self.current_num_clients - len(self.clients)
            # print("Num clients to readd", num_clients_to_readd)

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

    
    def initialize_clients_location(self):
        new_client_list = []
        locations = np.random.randint( self.left_road_limit, self.right_road_limit, size = len(self.clients))
        # if self.option['mean_velocity'] != 0:
        #     velocities_absolute = np.random.randint( self.mean_velocity - self.std_velocity, self.mean_velocity + self.std_velocity, size = len(self.clients))
        #     velocities_direction = np.array([random.choice([-1,1]) for i in range(len(self.clients))])
        #     velocities = velocities_absolute * velocities_direction
        #     # print(velocities_direction, velocities)
        # else:
        #     velocities = np.array([0 for i in range(len(self.clients))])

        for i  in range(len(self.clients)):
            client = self.clients[i]
            client.location = locations[i] 
            # client.velocity = velocities[i]
            # new_client_list.append(client)
        
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


    def unpack(self, packages_received_from_clients):
        """
        Unpack the information from the received packages. Return models and losses as default.
        :param
            packages_received_from_clients:
        :return:
            models: a list of the locally improved model
            losses: a list of the losses of the global model on each training dataset
        """
        models = [cp["model"] for cp in packages_received_from_clients]
        train_losses = [cp["train_loss"] for cp in packages_received_from_clients]
        valid_losses = [cp["valid_loss"] for cp in packages_received_from_clients]
        train_acc = [cp["train_acc"] for cp in packages_received_from_clients]
        valid_acc = [cp["valid_acc"] for cp in packages_received_from_clients]

        return models, (train_losses, valid_losses, train_acc, valid_acc)


class BasicEdgeServer(BasicServer):
    def __init__(self, option,model,cover_area, name = '', clients = [], test_data=None):
        super(BasicEdgeServer, self).__init__(option, model, clients, test_data)
        self.cover_area = cover_area
        self.name = name
        self.option = option
        self.total_datavol = 0
        # self.list_clients = []

    def unpack(self, packages_received_from_clients):
        """
        Unpack the information from the received packages. Return models and losses as default.
        :param
            packages_received_from_clients:
        :return:
            models: a list of the locally improved model
            losses: a list of the losses of the global model on each training dataset
        """
        models = [cp["model"] for cp in packages_received_from_clients]
        train_losses = [cp["train_loss"] for cp in packages_received_from_clients]
        valid_losses = [cp["valid_loss"] for cp in packages_received_from_clients]
        train_acc = [cp["train_acc"] for cp in packages_received_from_clients]
        valid_acc = [cp["valid_acc"] for cp in packages_received_from_clients]

        return models, (train_losses, valid_losses, train_acc, valid_acc)



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

    def train(self, model):
        """
        Standard local training procedure. Train the transmitted model with local training dataset.
        :param
            model: the global model
        :return
        """
        model.train()
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.epochs):
            for batch_id, batch_data in enumerate(data_loader):
                model.zero_grad()
                loss = self.calculator.get_loss(model, batch_data)
                loss.backward()
                optimizer.step()
        return

    def test(self, model, dataflag='valid'):
        """
        Evaluate the model with local data (e.g. training data or validating data).
        :param
            model:
            dataflag: choose the dataset to be evaluated on
        :return:
            eval_metric: task specified evaluation metric
            loss: task specified loss
        """
        dataset = self.train_data if dataflag=='train' else self.valid_data
        model.eval()
        loss = 0
        eval_metric = 0
        data_loader = self.calculator.get_data_loader(dataset, batch_size=64)
        for batch_id, batch_data in enumerate(data_loader):

            bmean_eval_metric, bmean_loss = self.calculator.test(model, batch_data)
            loss += bmean_loss * len(batch_data[1])
            eval_metric += bmean_eval_metric * len(batch_data[1])
        eval_metric =1.0 * eval_metric / len(dataset)
        loss = 1.0 * loss / len(dataset)
        return eval_metric, loss

    def unpack(self, received_pkg):
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
        model = self.unpack(svr_pkg)
        # print("CLient unpacked to package")
        train_loss = self.train_loss(model)
        valid_loss = self.valid_loss(model)
        train_acc = self.train_metrics(model)
        valid_acc = self.valid_metrics(model)

        # print("Client evaluated the train losss")
        self.train(model)
        # print("Client trained the model")
        eval_dict = {'train_loss': train_loss, 
                      'valid_loss': valid_loss,
                      'train_acc':train_acc,
                      'valid_acc': valid_acc}
        cpkg = self.pack(model, eval_dict)
        # print("Client packed and finished")
        return cpkg

    def pack(self, model,eval_dict ):
        """
        Packing the package to be send to the server. The operations of compression
        of encryption of the package should be done here.
        :param
            model: the locally trained model
            loss: the loss of the global model on the local training dataset
        :return
            package: a dict that contains the necessary information for the server
        """
        pkg = {'model': model} | eval_dict
        return pkg


    def train_loss(self, model):
        """
        Get the task specified loss of the model on local training data
        :param model:
        :return:
        """
        return self.test(model,'train')[1]

    def train_metrics(self, model):
        """
        Get the task specified metrics of the model on local training data
        :param model:
        :return:
        """
        return self.test(model,'train')[0]

    def valid_loss(self, model):
        """
        Get the task specified loss of the model on local validating data
        :param model:
        :return:
        """
        return self.test(model)[1]

    def valid_metrics(self, model):
        """
        Get the task specified loss of the model on local validating data
        :param model:
        :return:
        """
        return self.test(model)[0]


