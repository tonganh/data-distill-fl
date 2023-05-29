from .fedbase import BasicServer, BasicClient
import numpy as np

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)

        
        self.all_clients = self.clients.copy()
        self.k_clients=  self.option['k_clients']
        self.active_clients = self.clients[:self.k_clients]
        self.reserved_clients = self.clients[self.k_clients :]
        self.num_clients_add = self.option['num_clients_add']
        self.num_clients_delete = self.option['num_clients_delete']

        self.rounds_per_add = self.option['rounds_per_add']
        self.rounds_per_delete = self.option['rounds_per_delete']
        self.clients = self.active_clients.copy()
        self.num_clients = len(self.clients)

        self.avg_client_train_losses = []
        self.avg_client_valid_losses = []
        self.avg_client_train_metrics = []
        self.avg_client_valid_metrics = []
        self.client_metrics = {}
        self.rounds_since_participated = {}
        for client in self.all_clients:
            self.client_metrics[client.name] = [['Round', 'train_losses', 'train_accs', 'val_Losses', 'val_accs']]
            self.rounds_since_participated[client.name] = -1

    def sample(self, t):
        """Sample the clients.
        :param
            replacement: sample with replacement or not
        :return
            a list of the ids of the selected clients
        """
        current_clients = [cid for cid in range(len(self.clients))]
        if  ((t % self.rounds_per_add == 0 and t >= 5) or t == 5):
            if len(self.reserved_clients) > self.num_clients_add:
                self.clients.extend(self.reserved_clients[:self.num_clients_add].copy())
                self.reserved_clients = self.reserved_clients[self.num_clients_add - 1:].copy()
            else:
                print('Can not add any more clients')
        if ((t % self.rounds_per_delete == 0 and t >= 5) or t == 5.0):
            if len(self.clients) >= 5:
                self.reserved_clients.extend(self.clients[:self.num_clients_delete].copy())
                self.clients = self.clients[self.num_clients_delete :].copy()
            else:
                print('Can not delete any more clients')
        self.num_clients = len(self.clients)

        all_clients = [cid for cid in range(self.num_clients)]
        selected_clients = []
        # collect all the active clients at this round and wait for at least one client is active and
        active_clients = []
        while(len(active_clients)<1):
            active_clients = [cid for cid in range(self.num_clients) if self.clients[cid].is_active()]
        # sample clients
        if self.sample_option == 'active':
            # select all the active clients without sampling
            selected_clients = active_clients
        if self.sample_option == 'uniform':
            # original sample proposed by fedavg
            selected_clients = list(np.random.choice(all_clients, self.clients_per_round, replace=False))
        elif self.sample_option =='md':
            # the default setting that is introduced by FedProx
            selected_clients = list(np.random.choice(all_clients, self.clients_per_round, replace=True, p=[nk / self.data_vol for nk in self.client_vols]))
        # drop the selected but inactive clients
        selected_clients = list(set(active_clients).intersection(selected_clients))
        print(f"Round {t} - Num clients: {len(selected_clients)} - List of clients : {selected_clients}")
        reserved_clients = [client.name for client in self.reserved_clients]
        print(reserved_clients)
        return selected_clients


    def iterate(self, t):
        """
        The standard iteration of each federated round that contains three
        necessary procedure in FL: client selection, communication and model aggregation.
        :param
            t: the number of current round
        """
        # sample clients: MD sampling as default but with replacement=False
        self.selected_clients = self.sample(t)
        # print("Done sampling")
        # training
        models, (client_names, train_losses, valid_losses, train_acc, valid_acc) = self.communicate(self.selected_clients)
        
        all_client_train_losses = []
        all_client_valid_losses = []
        all_client_train_metrics = []
        all_client_valid_metrics = []

        for i in range(len(client_names)):
            client_name = client_names[i]
            client_train_loss = train_losses[i]
            client_valid_loss = valid_losses[i]
            client_train_acc = train_acc[i]
            client_valid_acc = valid_acc[i]
            if client_name in self.client_metrics.keys():
                self.client_metrics[client_name].append([t,client_train_loss, client_train_acc, client_valid_loss, client_valid_acc])
            else:
                self.client_metrics[client_name] = [['Round','train_losses', 'train_accs', 'val_Losses', 'val_accs']]

            all_client_train_losses.append(client_train_loss)
            all_client_valid_losses.append(client_valid_loss)
            all_client_train_metrics.append(client_train_acc)
            all_client_valid_metrics.append(client_valid_acc)
        
        self.avg_client_train_losses.append(sum(all_client_train_losses) / len(all_client_train_losses))
        self.avg_client_valid_losses.append(sum(all_client_valid_losses) / len(all_client_valid_losses))
        self.avg_client_train_metrics.append(sum(all_client_train_metrics) / len(all_client_train_metrics))
        self.avg_client_valid_metrics.append(sum(all_client_valid_metrics) / len(all_client_valid_metrics))


        # Get clients not participating in this training round
        unused_clients = set(self.all_clients).difference(set(self.clients))
        for client in unused_clients:
            client_name = client.name
            if client_name in self.client_metrics.keys():
                self.client_metrics[client_name].append([t,-1,-1,-1, -1])
            else:
                self.client_metrics[client_name] = [['Round', 'train_losses', 'train_accs', 'val_Losses', 'val_accs']]



        # print("Done a training step")
        # check whether all the clients have dropped out, because the dropped clients will be deleted from self.selected_clients
        if not self.selected_clients: return
        # aggregate: pk = 1/K as default where K=len(selected_clients)
        self.model = self.aggregate(models, p = [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients])
        return

    def unpack(self, packages_received_from_clients):
        """
        Unpack the information from the received packages. Return models and losses as default.
        :param
            packages_received_from_clients:
        :return:
            models: a list of the locally improved model
            losses: a list of the losses of the global model on each training dataset
        """
        client_names = [cp['client_name']  for cp in packages_received_from_clients]
        models = [cp["model"] for cp in packages_received_from_clients]
        train_losses = [cp["train_loss"] for cp in packages_received_from_clients]
        valid_losses = [cp["valid_loss"] for cp in packages_received_from_clients]
        train_acc = [cp["train_acc"] for cp in packages_received_from_clients]
        valid_acc = [cp["valid_acc"] for cp in packages_received_from_clients]

        return models, (client_names, train_losses, valid_losses, train_acc, valid_acc)

class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
    
    
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
        pkg = {'client_name' : self.name,'model': model} | eval_dict
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


