import utils.fflow_distill as flw
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
                "total_transfer_size":[],
            }
        if "mp_" in server.name:
            test_metric, test_loss = server.test(device=torch.device('cuda:0'))
        else:
            test_metric, test_loss = server.test()
        # valid_metrics, valid_losses = server.test_on_clients(self.current_round, 'valid')
        # train_metrics, train_losses = server.test_on_clients(self.current_round, 'train')

        # print(len(valid_metrics), len(valid_losses))
        # print(len(train_metrics), len(train_losses))
        self.output['train_losses'] = server.avg_edge_train_losses
        self.output['valid_losses'] = server.avg_edge_valid_losses
        self.output['train_accs'] = server.avg_edge_train_metrics
        self.output['valid_accs'] = server.avg_edge_valid_metrics
        self.output['total_transfer_size'] = server.avg_edge_total_transfer_size
        
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
        print(self.temp.format("Total Transfer data:", self.output['total_transfer_size'][-1]))

        # dataset = server['task']
        if not os.path.exists('results_distill'.format(server.option['task'])):
            os.mkdir('results_distill'.format(server.option['task']))

        if not os.path.exists('results_distill/{}'.format(server.option['task'])):
            os.mkdir('results_distill/{}'.format(server.option['task']))


        path_save_data = 'results_distill/{}/remove_{}_ipc_{}_velocity_{}_edge_freq_{}'.format(
                                                                                server.option['remove_client'],
                                                                                server.option['task'],
                                                                                server.option['distill_ipc'], 
                                                                                server.option['mean_velocity'],
                                                                                server.option['edge_update_frequency'])
        
        if not os.path.exists(path_save_data):
            os.mkdir(path_save_data)
    
        process_option_setting_path_save = f'{path_save_data}/setting.json'
        with open(process_option_setting_path_save, 'w') as file:
            json.dump(server.option, file)


        test_results_path = '{}/test_results.csv'.format(path_save_data)
        
        experiment_df = pd.DataFrame(columns=['round','test_acc','test_loss','train_loss','val_loss','train_acc', 'val_acc','total_transfer_size'])
        experiment_df['round'] = [i for i in range(len(self.output['test_accs']))]
        experiment_df['test_acc'] = self.output['test_accs']
        experiment_df['test_loss'] = self.output['test_losses']
        experiment_df['train_loss'] = self.output['train_losses']
        experiment_df['val_loss'] = self.output['valid_losses']
        experiment_df['val_acc'] = self.output['valid_accs']
        experiment_df['train_acc'] =  self.output['train_accs']
        experiment_df['total_transfer_size'] =  self.output['total_transfer_size']
        
        experiment_df.to_csv(test_results_path,index=False)

        task_name = server.option['task']
        for edge in server.edges:
            edge_name = edge.name
            ipc = server.option['distill_ipc']
            client_train_results = server.edge_metrics[edge.name]
            client_df = pd.DataFrame(client_train_results)
            client_save_path = f'{path_save_data}/{edge_name}.csv'
            client_df.to_csv(client_save_path, index=False)




logger = MyLogger()

def main_distill():
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
    if server.option['distill_before_train']:
        #start distillation
        server.distill()
    # start federated optimization
    server.run()

if __name__ == '__main__':
    main_distill()




