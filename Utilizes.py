import torch
from torch.autograd import Variable
   
#==========================================================
#            Initial Strategy of Network
#==========================================================
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.1)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0.1)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.1)
        m.bias.data.fill_(0.1)  

#==========================================================
#       Compute Accuracy
#==========================================================
def classification_accuracy(e,c,data_loader):
    with torch.no_grad():
        correct = 0
        for batch_idx, (X, target) in enumerate(data_loader):
            X, target = Variable(X), Variable(target).long().squeeze()
            X, target = X.cuda(), target.cuda()
            t_fe = e.forward(X)

            output, _, _ = c(t_fe)           
            
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()
        
        return correct.item() / len(data_loader.dataset)    

#==========================================================
#       Grassmannian Manifold Metric
#==========================================================
def grassmann_dist_Fast(input1, input2):
    fea_dim = input1.shape[1]
    
    h_src = input1 - torch.mean(input1, dim=0)
    h_trg = input2 - torch.mean(input2, dim=0)
    
    _, D1, V1 = torch.svd(h_src)
    _, D2, V2 = torch.svd(h_trg)

    return torch.sum(torch.pow(V1.mm(V1.t())-V2.mm(V2.t()), 2))/(fea_dim*fea_dim), D1, D2

#==========================================================
#       Source Inter-class Similarity
#==========================================================
def Source_InterClass_sim_loss(h_s, target, source_Tmean, Sim_type = 'sum'):    
    uni_tar = target.unique()
    num_sam_class = torch.zeros(uni_tar.shape[0]).cuda()
    Class_mean = torch.zeros(uni_tar.shape[0],h_s.shape[1]).cuda()
    for i in range(uni_tar.shape[0]):
        Index_i = (target == uni_tar[i])
        num_sam_class[i] = Index_i.sum()
        Class_mean[i,:] = h_s[Index_i,:].mean(0)
           
    Class_mean = Class_mean - source_Tmean.repeat(Class_mean.shape[0],1)
    norm_CM = Class_mean.pow(2).data.sum(1).pow(1/2).unsqueeze(1)
    Class_mean = Class_mean.mul(1/norm_CM)
    
    SIM = 0
    
    if Sim_type == 'adj':
        for i in range(uni_tar.shape[0] - 1):
            for j in range(i+1, uni_tar.shape[0]):
                SIM += (Class_mean[i].mul(Class_mean[j]).sum()/2 + 1/2).pow(2)
        return SIM/(target.shape[0]*(target.shape[0]-1)/2)
    elif Sim_type == 'none':
        for i in range(uni_tar.shape[0] - 1):
            for j in range(i+1, uni_tar.shape[0]):
                SIM += Class_mean[i].mul(Class_mean[j]).sum().pow(2)
        return SIM/(target.shape[0]*(target.shape[0]-1)/2)
    elif Sim_type == 'sum':
        for i in range(uni_tar.shape[0] - 1):
            for j in range(i+1, uni_tar.shape[0]):
                SIM += Class_mean[i].mul(Class_mean[j]).sum() + 1/2
        return SIM/(target.shape[0]*(target.shape[0]-1)/2)

#==========================================================
#       Target Intra-class Similarity
#==========================================================  
def Target_IntraClass_sim_loss(h_t, pred_t, source_mean, Top_n = 0):
    norm_h = h_t.pow(2).data.sum(1).pow(1/2).unsqueeze(1)
    norm_s = source_mean.pow(2).data.sum(1).pow(1/2).unsqueeze(1)
    h_t = h_t.mul(1/norm_h)
    source_mean = source_mean.mul(1/norm_s)
    
    Flag = torch.ones(pred_t.shape).cuda()
    if Top_n:
        _, De_index = pred_t.sort(1,descending = True)
        for i in range(pred_t.shape[0]):
            Flag[i,De_index[i,Top_n:]] = 0    
    Sim = h_t.mm(source_mean.t()).mul(pred_t).mul(Flag)

    return -Sim.norm('fro').pow(2)/(pred_t.shape[0]*pred_t.shape[1]) # Square F-norm of weighted intra-class scatter