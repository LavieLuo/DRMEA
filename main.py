import torch
import argparse
import os
import os.path as osp
import numpy as np
import torch.nn as nn
from datetime import datetime
from torch.autograd import Variable
import torch.optim as optim
import warnings
from torch.utils import model_zoo
warnings.filterwarnings("ignore")

#===========
import network
import Utilizes
import Data

#==============================================================================
#                           Train Function
#==============================================================================
def train(config):
    # ================== Pass the Config
    dset_name = config['dset']
    source_domain_set, target_domain_set, save_name, n_classes  = \
        Data.Get_Domain_Meta(dset_name)
    Exptimes = int(config['exp_time'])
    num_tasks = len(source_domain_set)
    batch_size = int(config['bs'])
    epochs = int(config['maxEpo'])
   
    # ========== Hyperparameter of DRMEA ========
    Manifold_dim = [1024,512]
    Aligned_step = 0 # Apply Grass alignment at x-th epoch 
    PLGE_step = 10 # Apply target intra-class loss at x-th epoch 
    PLGE_Inter_step = 1 # Apply source inter-class loss at x-th epoch 
    PLGE_lambda_L2, PLGE_lambda_L1 = 1e1, 1e0 # \lambda_1
    Grass_lambda = 5e3 # \lambda_2
    Top_n = 1 # Top-k preserving scheme
    # ===========================================
    
    # ========= Downlaod Pretrained Model
    url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'    
    pretrained_dict = model_zoo.load_url(url,model_dir='./pretrained_dict')
    del pretrained_dict['fc.bias']
    del pretrained_dict['fc.weight']
    
    # ================== Training
    ACC_Recorder = np.zeros((Exptimes,num_tasks))  # Result recorder  
    Total_Result = np.zeros((3,epochs,Exptimes))
    for Domain_iter in range(num_tasks):
        source_domain = source_domain_set[Domain_iter]
        target_domain = target_domain_set[Domain_iter]
        source_loader = Data.data_loader(dset_name, source_domain, batch_size)
        target_loader = Data.data_loader(dset_name, target_domain, batch_size)        
        
        # Random Experiment
        Exp_iter = 0
        best_acc = 0
        while Exp_iter < Exptimes:
            # ============ Define Network Architecture
            if config['Network'] == 'ResNet50':
                e = network.ResNet50_Feature(network.Bottleneck, [3, 4, 6, 3])
                fea_dim = 2048
            c = network.C(fea_dim, Manifold_dim[0], Manifold_dim[1], n_classes)
            e.cuda()
            c.cuda()
            e.load_state_dict(pretrained_dict)
            c.apply(Utilizes.weights_init)
            
            # ============= Define Optimizer
            lr = 2e-4
            beta1=0.9
            beta2=0.999
            Optimizer = optim.Adam([{'params':e.parameters(), 'lr': lr*0.1},
                                    {'params':c.parameters()}],
                                    lr*1.5, [beta1, beta2], weight_decay=0.01)
            criterionQ_dis = nn.NLLLoss().cuda() 

            # ============= Training Epoches
            result = np.zeros((3,epochs))
            Exp_start_time = datetime.now()
            print("************ %1s→%1s:  %1s Start Experiment %1s training ************"%(source_domain,target_domain,Exp_start_time,Exp_iter+1))              
            for step in range(epochs):
                epoch_time_start = datetime.now()              
                #=================== Initialize the mean vector ========
                H_mean_update_L2 = torch.zeros(n_classes,512).cuda()
                C_num_count = torch.zeros(n_classes,1).cuda()
                H_mean_update_L1 = torch.zeros(n_classes,1024).cuda()
                #=======================================================                
                Current_loss = np.array([0])
                Current_Coral_Grass_loss_L1 = np.array([0])
                Current_Coral_Grass_loss_L2 = np.array([0])
                Current_PLGE_loss_L2 = np.array([0])
                Current_PLGE_loss_L1 = np.array([0])
                Current_PLGE_inter_loss_L2 = np.array([0])
                Current_PLGE_inter_loss_L1 = np.array([0])
                                
                for (X, target), (X_t, target_test) in zip(source_loader,target_loader):                
                    e.train()
                    c.train()
                    
                    X, target = Variable(X), Variable(target)
                    X, target = X.cuda(), target.cuda()
                    X_t = Variable(X_t)
                    X_t = X_t.cuda()
                    
                    # Init gradients
                    e.zero_grad()
                    c.zero_grad()
                    
                    s_fe = e.forward(X)
                    s_fe_t = e.forward(X_t)
                    pred_s, h_s, h_s2 = c(s_fe)
                    pred_t, h_t, h_t2 = c(s_fe_t)

                    if step >= (PLGE_step - 1):
                        # =============== compute the class mean vector ==========
                        sam_count = 0
                        Tensor_size = target.shape
                        if Tensor_size:
                            target = target
                        else:
                            target = target.unsqueeze(0)
                            
                        for i in target:
                            C_num_count[i] += 1
                            H_mean_update_L2[i,:] += h_s2[sam_count,:].data
                            H_mean_update_L1[i,:] += h_s[sam_count,:].data
                            sam_count += 1
                        # =========================================================
                    
                    #==========================================================
                    #                     Loss Part
                    #==========================================================
                    CE_loss = criterionQ_dis(torch.log(pred_s+1e-4), target)
                    
                    #===================== Align Loss =========================
                    if step <= (Aligned_step - 1):
                        Coral_Grass_loss_L1 = torch.zeros(1).squeeze(0).cuda()
                        Coral_Grass_loss_L2 = torch.zeros(1).squeeze(0).cuda()
                    else:
                        Coral_Grass_loss_L1, _, _ = Utilizes.grassmann_dist_Fast(h_s, h_t)
                        Coral_Grass_loss_L2, _, _ = Utilizes.grassmann_dist_Fast(h_s2, h_t2)
                    Align_loss = Grass_lambda*Coral_Grass_loss_L1 + Grass_lambda*Coral_Grass_loss_L2 
                    #===================== Align Loss =========================
                    
                    #================ Source Discriminative Loss ==============
                    if step <= (PLGE_Inter_step - 1):
                        PLGE_inter_loss_L2 = torch.zeros(1).squeeze(0).cuda()
                        PLGE_inter_loss_L1 = torch.zeros(1).squeeze(0).cuda()
                    else:
                        PLGE_inter_loss_L2 = PLGE_lambda_L2*Utilizes.Source_InterClass_sim_loss(h_s2, target, H_Tmean_use_L2, 'adj')
                        PLGE_inter_loss_L1 = PLGE_lambda_L1*Utilizes.Source_InterClass_sim_loss(h_s, target, H_Tmean_use_L1, 'adj')
                    Source_Discri_loss = PLGE_inter_loss_L2 + PLGE_inter_loss_L1 
                    #================ Source Discriminative Loss ==============
                    
                    #================ Target Discriminative Loss ==============
                    if step <= (PLGE_step - 1):
                        c_loss = CE_loss + Align_loss + Source_Discri_loss 
                    else:                            
                        PLGE_loss_L2 = PLGE_lambda_L2*Utilizes.Target_IntraClass_sim_loss(h_t2, pred_t, H_mean_use_L2, Top_n)
                        PLGE_loss_L1 = PLGE_lambda_L1*Utilizes.Target_IntraClass_sim_loss(h_t, pred_t, H_mean_use_L1, Top_n)
                        Target_Discri_loss = PLGE_loss_L2 + PLGE_loss_L1 
                        c_loss = CE_loss + Align_loss + Target_Discri_loss + Source_Discri_loss 
                    #================ Target Discriminative Loss ==============
                                                           
                    Current_loss = np.concatenate((Current_loss,c_loss.cpu().detach().numpy()[np.newaxis]),axis = 0)            
                    Current_Coral_Grass_loss_L1 = np.concatenate((Current_Coral_Grass_loss_L1,Coral_Grass_loss_L1.cpu().detach().numpy()[np.newaxis]),axis = 0)
                    Current_Coral_Grass_loss_L2 = np.concatenate((Current_Coral_Grass_loss_L2,Coral_Grass_loss_L2.cpu().detach().numpy()[np.newaxis]),axis = 0)
                    Current_PLGE_inter_loss_L2 = np.concatenate((Current_PLGE_inter_loss_L2,PLGE_inter_loss_L2.cpu().detach().numpy()[np.newaxis]),axis = 0)
                    Current_PLGE_inter_loss_L1 = np.concatenate((Current_PLGE_inter_loss_L1,PLGE_inter_loss_L1.cpu().detach().numpy()[np.newaxis]),axis = 0)
                    if step > (PLGE_step - 1):
                        Current_PLGE_loss_L2 = np.concatenate((Current_PLGE_loss_L2,PLGE_loss_L2.cpu().detach().numpy()[np.newaxis]),axis = 0)
                        Current_PLGE_loss_L1 = np.concatenate((Current_PLGE_loss_L1,PLGE_loss_L1.cpu().detach().numpy()[np.newaxis]),axis = 0)

                    c_loss.backward()
                    Optimizer.step()
                    
                
                
                H_mean_use_L2 = H_mean_update_L2.mul(1/C_num_count) # Class mean matrix
                H_Tmean_use_L2 = H_mean_update_L2.mean(0) # Total mean vector
                H_mean_use_L1 = H_mean_update_L1.mul(1/C_num_count)
                H_Tmean_use_L1 = H_mean_update_L1.mean(0)
                del H_mean_update_L2, H_mean_update_L1
                e.eval()
                c.eval()
                Test_start_time = datetime.now()
                print('========================== %1s | Testing start! ==========================='%(Test_start_time))
                source_acc = Utilizes.classification_accuracy(e,c,source_loader)
                target_acc = Utilizes.classification_accuracy(e,c,target_loader)
                
                Current_Coral_Grass_loss_L1 = np.sum(Current_Coral_Grass_loss_L1)/(Current_Coral_Grass_loss_L1.size - 1)
                Current_Coral_Grass_loss_L2 = np.sum(Current_Coral_Grass_loss_L2)/(Current_Coral_Grass_loss_L2.size - 1)
                Current_PLGE_inter_loss_L2 = np.sum(Current_PLGE_inter_loss_L2)/(Current_PLGE_inter_loss_L2.size - 1)
                Current_PLGE_inter_loss_L1 = np.sum(Current_PLGE_inter_loss_L1)/(Current_PLGE_inter_loss_L1.size - 1)
                if step > (PLGE_step - 1):
                    Current_PLGE_loss_L2 = np.sum(Current_PLGE_loss_L2)/(Current_PLGE_loss_L2.size - 1)
                    Current_PLGE_loss_L1 = np.sum(Current_PLGE_loss_L1)/(Current_PLGE_loss_L1.size - 1)
                Current_loss = np.sum(Current_loss)/(Current_loss.size - 1)
                
                result[:,step] = [target_acc,source_acc,Current_loss]
                #====================== Time =====================
                epoch_time_end = datetime.now()
                seconds = (epoch_time_end - epoch_time_start).seconds
                minutes = seconds//60
                second = seconds%60
                hours = minutes//60
                minute = minutes%60
                print('Source accuracy: {}'.format(source_acc)) 
                print('Target accuracy: {}'.format(target_acc)) 
                print('Max Target Accuracy: {}'.format(max(result[0,:])))
                print('Total_Loss: {}'.format(Current_loss))
                print('Current epoch time cost (including test): %1s Hour %1s'\
                      ' Minutes %1s Seconds'%(hours,minute,second))
                #========= Save the best model and write log
                if target_acc > best_acc:
                    best_acc = target_acc
                    torch.save({
                    'Epoch': (step+1),
                    'state_dict_Backbone': e.state_dict(),
                    'state_dict_Manifold': c.state_dict(),
                    'Manifold_Dim': [1024,512],
                    'best_prec1': best_acc,
                    }, config["output_path"]+'/BestModel_'+save_name[Domain_iter]+'.tar')
                log_str = 'Experiment: {:05d}, Epoch: {:05d}, test precision:'\
                ' {:.5f}'.format(Exp_iter+1, step+1, target_acc)
                config["out_file"].write(save_name[Domain_iter]+'||'+log_str+'\n')
                config["out_file"].flush()
                #=========== If target accuracy reach 1, start new experiment
                if max(result[0,:]) == 1:
                    print('Reach accuracy {1} at Epoch %1s !'%(step+1))
                    break
            #============== End this Experiment ==============
            seconds = (epoch_time_end - Exp_start_time).seconds
            minutes = seconds//60
            second = seconds%60
            hours = minutes//60
            minute = minutes%60
            Total_Result[:,:,Exp_iter] = result
            ACC_Recorder[Exp_iter,Domain_iter] = max(result[0,:])
            print('Starting TIme: {}'.format(Exp_start_time))
            print("Finishing TIme: {}".format(epoch_time_end))
            print('Total TIme Cost: %1s Hour %1s Minutes %1s Seconds'%(hours,minute,second))
            print("************ %1s→%1s: %1s End Experiment %1s training ************"%(source_domain,target_domain,epoch_time_end,Exp_iter+1))
            Exp_iter += 1
        #================= End this domain transfer task

                    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Discriminative Manifold Embedding and Alignment AAAI-2020')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='ResNet50', choices=['ResNet50'])
    parser.add_argument('--dset', type=str, default='ImageCLEF', choices=['ImageCLEF'])
    parser.add_argument('--mEpo', type=str, nargs='?', default='50', help='Max epoches')
    parser.add_argument('--ExpTime', type=str, nargs='?', default='10', help='Numbers of random experiments')
    parser.add_argument('--BatchSize', type=str, nargs='?', default='32', help='Mini-Batch size')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    config = {}
    config['gpu'] = args.gpu_id
    config["output_path"] = "Model_Log/" + args.dset
    config['exp_time'] = args.ExpTime
    config['bs'] = args.BatchSize
    config['maxEpo'] = args.mEpo
    config['Network'] = args.net
    config['dset'] = args.dset
    if not osp.exists(config["output_path"]):
        os.system('mkdir -p '+config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")
    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])
        
    train(config)

    
    
