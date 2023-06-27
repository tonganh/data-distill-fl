import utils.fflow_experiments as flw
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import os
import multiprocessing
import pandas as pd
import json

class MyLogger(flw.Logger):
    def log(self, server=None):
        if server==None: return
        if self.output == {}:
            self.output = {
                "meta":server.option,
                "mean_curve":[],
                "var_curve":[],
                "train_losses":[],
                'valid_losses': [],
                'train_accs': [],
                "test_accs":[],
                "test_losses":[],
                "valid_accs":[],
                "client_accs":{},
                "mean_valid_accs":[],
            }
        if "mp_" in server.name:
            test_metric, test_loss = server.test(device=torch.device('cuda:0'))
        else:
            test_metric, test_loss = server.test()
        # valid_metrics, valid_losses = server.test_on_clients(self.current_round, 'valid')
        # train_metrics, train_losses = server.test_on_clients(self.current_round, 'train')

        # print(len(valid_metrics), len(valid_losses))
        # print(len(train_metrics), len(train_losses))

        self.output['train_losses'] = server.avg_client_train_losses
        self.output['valid_losses'] = server.avg_client_valid_losses
        self.output['train_accs'] = server.avg_client_train_metrics
        self.output['valid_accs'] = server.avg_client_valid_metrics
        self.output['test_accs'].append(test_metric)
        self.output['test_losses'].append(test_loss)
        # self.output['mean_valid_accs'].append(sum([acc for acc in valid_metrics]) / len([acc for acc in valid_metrics]))        
        # self.output['mean_curve'].append(np.mean(valid_metrics))
        # self.output['var_curve'].append(np.std(valid_metrics))
        # for cid in range(server.num_clients):
        #     self.output['client_accs'][server.clients[cid].name]=[self.output['valid_accs'][i][cid] for i in range(len(self.output['valid_accs']))]
        print(self.temp.format("Training Loss:", self.output['train_losses'][-1]))
        print(self.temp.format("Validation Loss:", self.output['valid_losses'][-1]))
        print(self.temp.format("Testing Loss:", self.output['test_losses'][-1]))
        print(self.temp.format("Training Accuracy:", self.output['train_accs'][-1]))
        print(self.temp.format("Validating Accuracy:", self.output['valid_accs'][-1]))
        print(self.temp.format("Testing Accuracy:", self.output['test_accs'][-1]))

        # dataset = server['task']
        if not os.path.exists('results_experiments_1'.format(server.option['task'])):
            os.mkdir('results_experiments_1'.format(server.option['task']))

        if not os.path.exists('results_experiments_1/{}'.format(server.option['task'])):
            os.mkdir('results_experiments_1/{}'.format(server.option['task']))

        if not os.path.exists('results_experiments_1/{}/add_{}_delete_{}'.format(server.option['task'],
                                                                                server.option['num_clients_add'],
                                                                                server.option['num_clients_delete'])):
            os.mkdir('results_experiments_1/{}/add_{}_delete_{}'.format(server.option['task'], 
                                                                      server.option['num_clients_add'],
                                                                        server.option['num_clients_delete']))

        test_results_path = 'results_experiments_1/{}/add_{}_delete_{}/test_results.csv'.format(server.option['task'], 
                                                                                server.option['num_clients_add'],
                                                                                server.option['num_clients_delete'])
    
        
        experiment_df = pd.DataFrame(columns=['round','test_acc','test_loss','train_loss','val_loss','train_acc', 'val_acc'])
        experiment_df['round'] = [i for i in range(len(self.output['test_accs']))]
        experiment_df['test_acc'] = self.output['test_accs']
        experiment_df['test_loss'] = self.output['test_losses']
        experiment_df['train_loss'] = self.output['train_losses']
        experiment_df['val_loss'] = self.output['valid_losses']
        experiment_df['val_acc'] = self.output['valid_accs']
        experiment_df['train_acc'] =  self.output['train_accs']
        experiment_df.to_csv(test_results_path,index=False)

        task_name= server.option['task']
        num_add = server.option['num_clients_add']
        num_delete = server.option['num_clients_delete']
        for client in server.all_clients:
            client_name = client.name
            client_train_results = server.client_metrics[client.name]
            client_df = pd.DataFrame(client_train_results)
            client_save_path = f'results_experiments_1/{task_name}/add_{num_add}_delete_{num_delete}/{client_name}.csv'
            client_df.to_csv(client_save_path, index=False)
        print(1)
        self.output['train_loss_dict'] = server.train_loss_tracker
        with open(f'results_experiments_1/{task_name}/add_{num_add}_delete_{num_delete}/train_loss_progress.json','w') as f:
            json.dump(server.train_loss_tracker, f)

        self.output['valid_acc_dict'] = server.valid_acc_tracker
        with open(f'results_experiments_1/{task_name}/add_{num_add}_delete_{num_delete}/valid_acc_progress.json','w') as f:
            json.dump(server.valid_acc_tracker, f)

        self.output['test_acc_dict'] = server.test_acc_tracker
        with open(f'results_experiments_1/{task_name}/add_{num_add}_delete_{num_delete}/test_acc_progress.json','w') as f:
            json.dump(server.test_acc_tracker, f)





logger = MyLogger()

def main_experiments():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print("CUDA Available" ,torch.cuda.is_available())
    multiprocessing.set_start_method('spawn')
    # read options
    option = flw.read_option()
    # os.environ['MASTER_ADDR'] = "localhost"
    # os.environ['MASTER_PORT'] = '8888'
    # os.environ['WORLD_SIZE'] = str(3)
    # set random seed
    flw.setup_seed(option['seed'])
    # initialize server
    server = flw.initialize(option)
    # start federated optimization
    server.run()

if __name__ == '__main__':
    main_experiments()




