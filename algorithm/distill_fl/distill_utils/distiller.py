import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from algorithm.distill_fl.distill_utils.utils import get_loops, get_dataset, get_network, get_eval_pool, custom_evaluate_synset, get_daparam, custom_match_loss, get_time, TensorDataset, custom_epoch, DiffAugment, ParamDiffAug

# import matplotlib.pyplot as plt 

class Distiller():
    def __init__(self,model = 'ConvNet',method = 'DC',ipc  = 2,eval_mode = "S",num_eval=5,iteration=1000,lr_img = 0.1 ,lr_net = 0.01,
                 batch_real = 64,batch_train = 64,init = 'noise',dis_metric = 'ours', device = 'cuda',
                 num_classes = 10, dataset = "MNIST", save_path = None):
        # self.dataset = dataset
        self.model = model

        self.ipc = ipc # images per class
        self.eval_mode = eval_mode #eval mode
        self.lr_img = lr_img #lr update synth
        self.iteration = iteration # lr training model
        self.num_eval = num_eval #
        self.lr_net = lr_net
        self.batch_real = batch_real #batchsize for real data
        self.batch_train = batch_train #  batch training data
        self.init = init # way to init the synth 
        
        self.dis_metric = dis_metric    

        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'    
        self.device = device
        self.eval_it_pool = np.arange(0, self.iteration+1, 70).tolist() if self.eval_mode == 'S' or self.eval_mode == 'SS' else [self.iteration] # The list of iterations when we evaluate models and record results.
        self.model_eval_pool = get_eval_pool(self.eval_mode, self.model, self.model)

        self.outer_loop, self.inner_loop = get_loops(self.ipc)
        self.num_classes = num_classes
        self.dataset = dataset
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)        


        # self.dsa_param = ParamDiffAug()
        # self.dsa = True if self.method == 'DSA' else False      

    def distill(self,x_train, y_train, x_test, y_test, additional_message = ""):
        s = 32
        dev = torch.device('cuda')
        torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))
        torch.cuda.empty_cache()
        x_train = x_train.to(self.device)
        x_test = x_test.to(self.device)
        y_train = y_train.to(self.device)
        y_test = y_test.to(self.device)

        print(y_train, y_test)
        # channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(dataset, data_path)
        mean, std = get_dataset(self.dataset, self.save_path)
        accs_all_exps = dict() # record performances of all experiments
        for key in self.model_eval_pool:
            accs_all_exps[key] = []

        data_save = []

        original_classes = np.unique(y_train.detach().cpu().numpy())
        original_classes = [int(label) for label in original_classes]
        self.num_classes = len(original_classes)
        classes = [i for i in range(self.num_classes)]
        class_mappings = {}
        reverse_class_mappings = {}
        for i in range(len(classes)):
            class_mappings[original_classes[i]] = classes[i]
            reverse_class_mappings[classes[i]] = original_classes[i]
        # print(original_classes, classes)
        # print(class_mappings, reverse_class_mappings)

        # print(self.num_classes)
        # ''' organize the real dataset '''
        # images_all = []
        # labels_all = []
        # indices_class = [[] for c in classes]
        indices_class = {}
        # images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        # labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        for i, lab in enumerate(y_train.detach().cpu().numpy().tolist()):
            transformed_class_name = class_mappings[int(lab)]
            if transformed_class_name in indices_class.keys():
                indices_class[transformed_class_name].append(i)
            else:
                indices_class[transformed_class_name]  = []
        # print(original_classes)
        # print(class_mappings)
        # print(indices_class)
        # images_all = torch.cat(images_all, dim=0).to(self.device)
        # labels_all = torch.tensor(labels_all, dtype=torch.long, device=self.device)


        for c in classes:
            print('class c = %d: %d real images'%(c, len(indices_class[c])))
        
        def get_images(c, n): # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return x_train[idx_shuffle]
        # print(y_train,y_train.shape)
        # print(y_test, y_test.shape)

        new_y_train = []
        for y in y_train.detach().cpu().numpy().tolist():
            transformed_y = class_mappings[y]
            new_y_train.append(transformed_y)
        
        y_train =  torch.from_numpy(np.array(new_y_train))
        new_y_test = []
        for y in y_test.detach().cpu().numpy().tolist():
            if y in class_mappings.keys():
                transformed_y = class_mappings[y]
                new_y_test.append(transformed_y)
            else:
                new_y_test.append(0)
        y_test = torch.from_numpy(np.array(new_y_test))
        # print(new_y_test)
        test_dataset  =  TensorDataset(x_test, y_test)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

        channel = x_train.shape[1]
        im_size = (x_train.shape[2], x_train.shape[3])
        
        # for ch in range(channel):
        #     print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(x_train[:, ch]), torch.std(x_train[:, ch])))


        ''' initialize the synthetic data '''
        image_syn = torch.randn(size=(self.num_classes*self.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=self.device)

        label_syn = torch.tensor([np.ones(self.ipc)*i for i in range(self.num_classes)], requires_grad=False, device=self.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

        if self.init == 'real':
            try:
                for c in range(self.num_classes):
                    image_syn.data[c*self.ipc:(c+1)*self.ipc] = get_images(c, self.ipc).detach().data
                print('initialize synthetic data from random real images')

            except:
                print('initialize synthetic data from random noise')

        else:
            print('initialize synthetic data from random noise')

        '''training'''
        optimizer_img = torch.optim.SGD([image_syn, ], lr=self.lr_img, momentum=0.5) # optimizer_img for synthetic data
        optimizer_img.zero_grad()
        criterion = nn.CrossEntropyLoss().to(self.device)
        print('%s training begins'%get_time())

        best_val_result = -999
        best_image_syn = None
        best_label_syn = None
        for it in range(self.iteration+1):
            if it in self.eval_it_pool:
                for model_eval in self.model_eval_pool:
                    print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(self.model, model_eval, it))
                    self.epoch_eval_train = 300

                    self.dc_aug_param = get_daparam(self.dataset, self.model, model_eval, self.ipc)
                    print('DC augmentation parameters: \n', self.dc_aug_param)
                    
                    accs = []

                    for it_eval in range(self.num_eval):
                        net_eval = get_network(model_eval,channel,self.num_classes, im_size).to(self.device)
                        image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach()) 

                    _, acc_train, acc_test = custom_evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, self.lr_net,self.device,self.epoch_eval_train,self.batch_train,self.dc_aug_param)
                    accs.append(acc_test)
                    print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))
                    if acc_test > best_val_result:
                        acc_test = best_val_result
                        print("Saved best distill data")
                        original_label_syn = []
                        for y in label_syn.detach().cpu().numpy().tolist():
                            original_label_syn.append(reverse_class_mappings[y])
                        original_label_syn = torch.from_numpy(np.array(original_label_syn))
                        torch.save(copy.deepcopy(image_syn.detach().cpu()), self.save_path + '_x_distill.pt')
                        torch.save(copy.deepcopy(original_label_syn.detach().cpu()), self.save_path + '_y_distill.pt')

                        best_image_syn = copy.deepcopy(image_syn.detach().cpu())
                        best_label_syn = copy.deepcopy(original_label_syn.detach().cpu())
                        save_name = os.path.join(self.save_path, 'vis_DS_%s_%s_%dipc.png'%( self.dataset, self.model, self.ipc))
                        image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                        for ch in range(channel):
                            image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
                        image_syn_vis[image_syn_vis<0] = 0.0
                        image_syn_vis[image_syn_vis>1] = 1.0
                        save_image(image_syn_vis, save_name, nrow=self.ipc) # Trying normalize = True/False may get better visual effects.

                        if it == self.iteration: # record the final results
                            accs_all_exps[model_eval] += accs
                

            ''' Train synthetic data '''
            net = get_network(self.model, channel, self.num_classes, im_size).to(self.device) # get a random model
            net.train()
            net_parameters = list(net.parameters())
            optimizer_net = torch.optim.SGD(net.parameters(), lr=self.lr_net)  # optimizer_img for synthetic data
            optimizer_net.zero_grad()
            loss_avg = 0
            self.dc_aug_param = None  # Mute the DC augmentation when learning synthetic data (in inner-loop epoch function) in oder to be consistent with DC paper.
            
            for ol in range(self.outer_loop):

                ''' freeze the running mu and sigma for BatchNorm layers '''
                # Synthetic data batch, e.g. only 1 image/batch, is too small to obtain stable mu and sigma.
                # So, we calculate and freeze mu and sigma for BatchNorm layer with real data batch ahead.
                # This would make the training with BatchNorm layers easier.
                BN_flag = False
                BNSizePC = 16  # for batch normalization
                for module in net.modules():
                    if 'BatchNorm' in module._get_name(): #BatchNorm
                        BN_flag = True
                if BN_flag:
                    img_real = torch.cat([get_images(c, BNSizePC) for c in range(self.num_classes)], dim=0)
                    net.train() # for updating the mu, sigma of BatchNorm
                    output_real = net(img_real) # get running mu, sigma
                    for module in net.modules():
                        if 'BatchNorm' in module._get_name():  #BatchNorm
                            module.eval() # fix mu and sigma of every BatchNorm layer
            
                ''' update synthetic data '''
                loss = torch.tensor(0.0).to(self.device)
                for c in classes:
                    img_real = get_images(c, self.batch_real)
                    if img_real.shape[0] > 0:
                        lab_real = torch.ones((img_real.shape[0],), device=self.device, dtype=torch.long) * c
                        img_syn = image_syn[c*self.ipc:(c+1)*self.ipc].reshape((self.ipc, channel, im_size[0], im_size[1]))
                        lab_syn = torch.ones((self.ipc,), device=self.device, dtype=torch.long) * c
                        output_real = net(img_real)
                        loss_real = criterion(output_real, lab_real)
                        gw_real = torch.autograd.grad(loss_real, net_parameters)
                        gw_real = list((_.detach().clone() for _ in gw_real))

                        output_syn = net(img_syn)
                        # print(self.device)
                        loss_syn = criterion(output_syn, lab_syn)
                        gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

                        loss += custom_match_loss(gw_syn, gw_real, self.device,self.dis_metric)

                optimizer_img.zero_grad()
                loss.backward()
                optimizer_img.step()
                loss_avg += loss.item()

                if ol == self.outer_loop - 1:
                    break


                ''' update network '''
                image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())  # avoid any unaware modification
                dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
                trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=self.batch_train, shuffle=True, num_workers=0)
                for il in range(self.inner_loop):
                    custom_epoch('train', trainloader, net, optimizer_net, criterion,False,self.device,self.dc_aug_param )
            
            # print(self.num_classes)
            # print(self.outer_loop)
            loss_avg /= (self.num_classes*self.outer_loop)

            if it%10 == 0:
                print('%s iter = %04d, loss = %.4f - ' % (get_time(), it, loss_avg), end =" ")
                print(f'{additional_message}')

            # if it == self.iteration: # only record the final results
            #     data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                # torch.save({'data': data_save }, os.path.join(save_path, 'res_DS_%s_%s_%dipc.pt'%( dataset, self.model, self.ipc)))
        return best_image_syn, best_label_syn
