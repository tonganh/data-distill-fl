import random
from typing import List
from utils import fmodule
import sys
sys.path.append('..')
from  algorithm.fedbase import BasicServer, BasicClient
import copy
import math
import numpy as np
import torch
from benchmark.toolkits import XYDataset
from sklearn.model_selection import train_test_split
from algorithm.distill_fl.distill_utils.distiller import Distiller
import os
from main_distill import logger

import datetime
import logging

now = datetime.datetime.now()
formatted_date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

directory_path = "log"  # specify the directory where log files will be saved
log_file_name = f'{directory_path}/log_kip_fedbase_{formatted_date_time}.log'
# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Create a file handler
file_handler = logging.FileHandler(log_file_name)
file_handler.setLevel(logging.DEBUG)

# Create a stream handler (for console output)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

# Create a formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Add the handlers to the logger
logger_anhtn = logging.getLogger('')
logger_anhtn.addHandler(file_handler)
logger_anhtn.addHandler(stream_handler)


class BasicCloudServer(BasicServer):
    def __init__(self, option, model ,clients,test_data = None):
        super(BasicCloudServer, self).__init__(option, model, clients, test_data)
        # self.clients = []
        self.model = model
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
        self.deleted_clients_name  = []

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


    def get_model_size(self, model):
        num_params = sum(p.numel() for p in model.parameters())
        # Mỗi tham số là một số thực dấu chấm động 32 bit, tức là 4 byte
        num_bytes = num_params * 4
        return num_bytes
        
    def distill(self):
        print("First, distill data from clients' side")
        for client in self.clients:
            client.distill_data()
    
    def get_clients_names(self):
        clients_name = [client.name for client in self.clients]
        return clients_name


    def delete_clients(self, count_remove_clients):
        clients_after_delete = random.sample(self.clients, k=len(self.clients)-count_remove_clients)
        clients_name_initial = [client.name for client in self.clients]
        clients_name_deleted = [client.name for client in clients_after_delete]
        
        # Tạo tập hợp các tên khách hàng duy nhất
        clients_name_initial_set = set(clients_name_initial)
        clients_name_deleted_set = set(clients_name_deleted)
        
        # Lấy danh sách các tên khách hàng đã bị xóa
        removed_clients_names = list(clients_name_initial_set - clients_name_deleted_set)
        
        print(f'Before delete, total clients is: {len(self.clients)}')
        self.clients = clients_after_delete
        print(f'After delete, total clients is: {len(self.clients)}')
        self.deleted_clients_name.extend(removed_clients_names)
        print(f'Name deleted client: {self.deleted_clients_name}')
        return removed_clients_names


    def run(self):
        """
        Start the federated learning symtem where the global model is trained iteratively.
        """
        logger.time_start('Total Time Cost')
        print("Load distilled data")
        for client in self.clients:
            print(f"Loaded client {client.name}")
            client.load_distill_data()

        for edge in self.edges:
            edge.model = copy.deepcopy(self.model)

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
        clients_names  = self.get_clients_names()
        if len(self.clients) < self.mean_num_clients - self.std_num_clients:
            self.current_num_clients = np.random.randint(low = self.mean_num_clients - self.std_num_clients, high=self.mean_num_clients+self.std_num_clients + 1,
                                                        size=1)[0]
            num_clients_to_readd = self.current_num_clients - len(self.clients)
            if num_clients_to_readd < len(self.unused_clients_queue):
                clients_to_readd = random.sample(self.unused_clients_queue, k = num_clients_to_readd)
                for client in clients_to_readd:
                    client.location =np.random.randint( self.left_road_limit, self.right_road_limit, size =1)[0]
                    client_name = client.name
                    if client_name not in clients_names and client_name not in self.deleted_clients_name:
                        self.clients.append(client)
                        clients_names.append(client_name)
                    # client.location = client
            else:
                clients_to_readd = self.unused_clients_queue
                for client in clients_to_readd:
                    client.location =np.random.randint( self.left_road_limit, self.right_road_limit, size =1)[0]
                    client_name = client.name
                    if client_name not in clients_names and client_name not in self.deleted_clients_name:
                        self.clients.append(client)
                        clients_names.append(client_name)
                        
        if len(self.clients) > 100:
            import pdb; pdb.set_trace()

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

        # listen for the client's response and return None if the client drops out
        # if self.clients[client_id].is_drop(): return None
        reply = edge.reply()
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

    def unpack(self, packages_received_from_edges):
        """
        Unpack the information from the received packages. Return models and losses as default.
        :param
            packages_received_from_clients:
        :return:
            models: a list of the locally improved model
            losses: a list of the losses of the global model on each training dataset
        """
        edge_names = [cp['edge_name']  for cp in packages_received_from_edges]
        models = [cp["model"] for cp in packages_received_from_edges]
        train_losses = [cp["train_loss"] for cp in packages_received_from_edges]
        valid_losses = [cp["valid_loss"] for cp in packages_received_from_edges]
        train_acc = [cp["train_acc"] for cp in packages_received_from_edges]
        valid_acc = [cp["valid_acc"] for cp in packages_received_from_edges]

        return models, (edge_names, train_losses, valid_losses, train_acc, valid_acc)



class BasicEdge(BasicClient):
    def __init__(self, option,model,cover_area, name = '', clients = [], test_data=None):
        super(BasicEdge, self).__init__(option, model, clients, test_data)
        self.cover_area = cover_area
        self.name = name
        self.option = option
        self.total_datavol = 0

        self.X_all = np.array([])
        self.y_all = np.array([])

        self.X_train  = np.array([])
        self.y_train = np.array([])
        self.X_valid = np.array([])
        self.y_valid = np.array([])

        self.datavol = None

        self.train_data = None
        self.valid_data = None
        self.clients_collected  = []
        self.total_transfer_size = 0
        # self.list_clients = []
    
    def split_data(self):
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.X_all, self.y_all, 
                                                                                 test_size = 0.2,random_state=self.option['seed'])
        # print("Split data")
        # print(self.X_train.shape, self.X_valid.shape, self.y_train.shape)
    def collect_distilled_data_from_client(self, clients: List[BasicClient]):
        # Update x_all and y_all by appending client's distill data, if client not already sent
        total_transfer_in_round = 0
        for client in clients:
            if client.name not in self.clients_collected:
                total_transfer_in_round   += client.total_size
                if self.X_all.size == 0:
                    self.X_all = client.x_distill
                    self.y_all = client.y_distill
                    # print(self.X_all.shape, self.y_all)
                else:
                    self.X_all = np.concatenate([self.X_all, client.x_distill],axis = 0)
                    self.y_all = np.concatenate([self.y_all, client.y_distill], axis = 0)
                    # print(self.X_all.shape, self.y_all)

                self.split_data()
                print('Client name debugging: ', client.name)
                print('Client transfer data: ', client.total_size)
                
                self.train_data = XYDataset(self.X_train, self.y_train, client_name=client.name)
                self.valid_data = XYDataset(self.X_valid, self.y_valid, client_name=client.name)
                self.clients_collected.append(client.name)
                self.datavol = self.X_train.shape[0]
        self.total_transfer_size = total_transfer_in_round

        
    def train(self):
        """
        Standard local training procedure. Train the transmitted model with local training dataset.
        :param
            model: the global model
        :return
        """
        self.model.train()
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, self.model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.epochs):
            for batch_id, batch_data in enumerate(data_loader):
                self.model.zero_grad()
                loss = self.calculator.get_loss(self.model, batch_data)
                loss.backward()
                optimizer.step()
        return

    def test(self, dataflag='valid'):
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
        self.model.eval()
        loss = 0
        eval_metric = 0
        data_loader = self.calculator.get_data_loader(dataset, batch_size=64)
        for batch_id, batch_data in enumerate(data_loader):

            bmean_eval_metric, bmean_loss = self.calculator.test(self.model, batch_data)
            loss += bmean_loss * len(batch_data[1])
            eval_metric += bmean_eval_metric * len(batch_data[1])
        eval_metric =1.0 * eval_metric / len(dataset)
        loss = 1.0 * loss / len(dataset)
        return eval_metric, loss

    def reply(self):
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
        # model = self.unpack(svr_pkg)
        # print("CLient unpacked to package")
        train_loss = self.train_loss()
        valid_loss = self.valid_loss()
        train_acc = self.train_metrics()
        valid_acc = self.valid_metrics()

        # print("Client evaluated the train losss")
        self.train()
        # print("Client trained the model")
        eval_dict = {'train_loss': train_loss, 
                      'valid_loss': valid_loss,
                      'train_acc':train_acc,
                      'valid_acc': valid_acc}
        cpkg = self.pack( eval_dict)
        # print("Client packed and finished")
        return cpkg


    def pack(self, eval_dict ):
        """
        Packing the package to be send to the server. The operations of compression
        of encryption of the package should be done here.
        :param
            model: the locally trained model
            loss: the loss of the global model on the local training dataset
        :return
            package: a dict that contains the necessary information for the server
        """
        pkg = {'edge_name' : self.name,'model': self.model} | eval_dict
        return pkg


    def train_loss(self):
        """
        Get the task specified loss of the model on local training data
        :param model:
        :return:
        """
        return self.test('train')[1]

    def train_metrics(self):
        """
        Get the task specified metrics of the model on local training data
        :param model:
        :return:
        """
        return self.test('train')[0]

    def valid_loss(self):
        """
        Get the task specified loss of the model on local validating data
        :param model:
        :return:
        """
        return self.test()[1]

    def valid_metrics(self):
        """
        Get the task specified loss of the model on local validating data
        :param model:
        :return:
        """
        return self.test()[0]




    # def unpack(self, packages_received_from_clients):
    #     """
    #     Unpack the information from the received packages. Return models and losses as default.
    #     :param
    #         packages_received_from_clients:
    #     :return:
    #         models: a list of the locally improved model
    #         losses: a list of the losses of the global model on each training dataset
    #     """
    #     models = [cp["model"] for cp in packages_received_from_clients]
    #     train_losses = [cp["train_loss"] for cp in packages_received_from_clients]
    #     valid_losses = [cp["valid_loss"] for cp in packages_received_from_clients]
    #     train_acc = [cp["train_acc"] for cp in packages_received_from_clients]
    #     valid_acc = [cp["valid_acc"] for cp in packages_received_from_clients]

    #     return models, (train_losses, valid_losses, train_acc, valid_acc)



    # def update_list_clients(self,clients):
    #     self.list_clients = clients

    

class BasicMobileClient(BasicClient):
    def __init__(self, option, location = 0, velocity = 0, name='', train_data=None, valid_data=None ):
        super(BasicMobileClient, self).__init__(option, name, train_data, valid_data)
        self.location = location
        self.velocity = velocity
        self.option = option
        self.x_distill = None
        self.y_distill = None

        
        if 'mnist' in self.option['task'] or 'cifar10' in self.option['task']:
            self.num_classes = 10
        elif  'cifar100' in self.option['task']:
            self.num_classes = 100
        
        self.distill_iters = self.option['distill_iters']
        self.task_name = self.option['task']
        
        distill_path_dataset = os.path.join( f'fedtask/{self.task_name}/', self.option['distill_data_path'])
        if not os.path.exists(distill_path_dataset):
            os.mkdir(distill_path_dataset)
            
        self.distill_save_path = os.path.join( f'fedtask/{self.task_name}/', self.option['distill_data_path'],str(self.option['distill_ipc']))

        if not os.path.exists(self.distill_save_path):
            os.mkdir(self.distill_save_path)      
        self.distill_save_path = os.path.join(self.distill_save_path,f'{self.name}/')
        print("Path", self.distill_save_path)
        # self.distill_save_path = os.path.join(self.distill_save_path, self.option['task'])  
        if not os.path.exists(self.distill_save_path):
            os.mkdir(self.distill_save_path)      
        
        if 'mnist' in self.option['task']:
            self.dataset =  'MNIST'
        elif 'cifar10' in self.option['task']:
            self.dataset  = 'CIFAR10'
        elif 'cifar100' in self.option['task']:
            self.dataset = 'CIFAR100'
        self.ipc = self.option['distill_ipc']
        self.distiller = Distiller(ipc=self.ipc, iteration=self.distill_iters, dataset = self.dataset, save_path = self.distill_save_path)
        # self.mean_velocity = mean_velocity
        # self.std_velocity = std_velocity
        # self.current_velocity = mean_velocity
    
    # def get_current_velocity(self):
    #     self.current_velocity = np.random.randint(low=self.mean_velocity - self.std_velocity, high=self.mean_velocity + self.std_velocity, size = 1)[0]

    def update_location(self):
        self.location += self.velocity

    def get_location(self):
        return self.location
    
    def distill_data(self):
        message = f"Distilling data from client: {self.name}"
        print(message)
        # import pdb; pdb.set_trace()
        x_train, y_train, x_val, y_val = self.train_data.X, self.train_data.Y, self.valid_data.X, self.valid_data.Y
        print(f'Client name: {self.name}')
        print(f'Check data class for each client. Client: {self.name}')
        print(set(y_train))
        logger_anhtn.info(f'Check data class for each client. Client: {self.name}')
        logger_anhtn.info(set(y_train))
        # print("Data from client"x_val,y_val)
        self.distiller.distill(X_TRAIN_RAW  = x_train, LABELS_TRAIN = y_train,X_TEST_RAW= x_val,
                               LABELS_TEST= y_val, additional_message=message)
    
    def load_distill_data(self):
        self.x_distill = torch.load(os.path.join(self.distill_save_path,'x_distill.pt')).detach().cpu().numpy()
        self.y_distill = torch.load(os.path.join(self.distill_save_path,'y_distill.pt')).detach().cpu().numpy()

        # print(self.x_distill.shape, self.y_distill)
        # print(type(self.x_distill), type(self.x_distill))



