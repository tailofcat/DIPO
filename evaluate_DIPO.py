import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

 
def M_dis(x, y, ver_cov):
    mx = x - y
    distance = torch.diag(torch.sqrt(mx @ ver_cov @ mx.t()))
    return distance
    

def iterative_density_driven(data, prototype, ver_cov, topk, n_ways):
    prototype_tensor = torch.Tensor(prototype).cuda()
    prob_matrix = torch.exp(-torch.stack([M_dis(data_tensor, prototype_tensor[i], ver_cov[i]) for i in range(n_ways)], dim=0))
    prob_matrix /= prob_matrix.sum(dim=0).unsqueeze(1).expand(-1, prob_matrix.shape[0]).t()
    sim_M, pred = prob_matrix.topk(topk, 1, True, True)
    topk_similar = data_tensor[pred,:]
    return topk_similar


if __name__ == '__main__':     
    # ---- data loading
    dataset = 'miniImagenet'
    n_shot = 1
    n_ways = 5
    n_queries = 15
    n_runs = 10000
    n_lsamples = n_ways * n_shot
    n_usamples = n_ways * n_queries
    n_samples = n_lsamples + n_usamples
         
    st, end, step = 5, 19, 1
    k = 64
    
    import FSLTask
    cfg = {'shot': n_shot, 'ways': n_ways, 'queries': n_queries}
    FSLTask.loadDataSet(dataset)
    FSLTask.setRandomStates(cfg)
    ndatas = FSLTask.GenerateRunSet(end=n_runs, cfg=cfg)
    ndatas = ndatas.permute(0, 2, 1, 3).reshape(n_runs, n_samples, -1)
    labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries, 5).clone().view(n_runs,
                                                                                                        n_samples)
    # ---- Base class statistics
    beta = 0.5
    base_mean = []
    base_cov = []
    base_features_path = "./checkpoints/%s/WideResNet28_10_S2M2_R/last/base_features.plk"%dataset
    with open(base_features_path, 'rb') as f:
        data = pickle.load(f)
        for key in data.keys():
            feature = np.array(data[key])
            mean = np.mean(feature, axis=0)
            feature = np.power(feature[:, ] ,beta)      
            cov = np.cov(feature.T)
            base_mean.append(mean)
            base_cov.append(cov)
            
    
    base_mean = torch.Tensor(np.array(base_mean)).cuda()
    base_cov = torch.Tensor(np.array(base_cov)).cuda()     
    
    # ---- classification for each task                                                         
    acc_list = []
    print('Start classification for %d tasks...'%(n_runs))
    for it in tqdm(range(n_runs)):
   
        support_data = ndatas[it][:n_lsamples].numpy()
        support_label = labels[it][:n_lsamples].numpy()
        query_data = ndatas[it][n_lsamples:n_lsamples+n_usamples].numpy()
        query_label = labels[it][n_lsamples:n_lsamples+n_usamples].numpy()
        row_support_set = [support_data[j::n_ways] for j in range(n_ways)]
        
        # ---- Tukey's transform
        support_data = np.power(support_data[:, ] ,beta)
        query_data = np.power(query_data[:, ] ,beta)

        # ---- Seeking the relevant base classes
        support_set = [support_data[j::n_ways] for j in range(n_ways)]
        _, dim = support_data.shape
        
        ver_cov_tensor = []
        for i in range(n_ways):
            prototype_tensor = torch.Tensor(np.array(row_support_set[i]))
            prototype_tensor = torch.mean(prototype_tensor, dim=0).unsqueeze(0)
            similar = torch.mm(F.normalize(prototype_tensor).cuda(), F.normalize(base_mean).cuda().transpose(0,1)) #a*b 1,38400
            sim_cos, pred = similar[0].topk(k, 0, True, True) 
            sim_weight = sim_cos / torch.sum(sim_cos) 
            c_cov = torch.sum(sim_weight.unsqueeze(1).unsqueeze(1)*base_cov[pred,:], dim=0)
            c_cov = c_cov + torch.trace(c_cov) / dim * torch.eye(dim).cuda()
            c_ver_cov = torch.linalg.inv(c_cov)
            ver_cov_tensor.append(c_ver_cov)
            
        # ---- Density-Driven Strategy
        support_set = torch.Tensor(np.array(support_set)).cuda()
        data_tensor = torch.Tensor(query_data).cuda()
        X_aug = support_set
        for tk in range(st, end, step):
            topk_similar = iterative_density_driven(data=data_tensor, prototype=torch.mean(X_aug, dim=1), ver_cov=ver_cov_tensor, topk=tk, n_ways=n_ways)
            X_aug = torch.cat((support_set, topk_similar), dim=1)
        
        
        # ---- Classification
        prototype_tensor = torch.mean(X_aug, dim=1)
        H = -torch.stack([M_dis(data_tensor, prototype_tensor[i], ver_cov_tensor[i]) for i in range(n_ways)], dim=0).t()
        _, pred = H.topk(1, 1, True, True)
        predicts = np.concatenate(support_label[pred.cpu().numpy()])
        
        acc = np.mean(predicts == query_label)
        acc_list.append(acc*100)
    print('%s %d way %d shot  ACC : %f'%(dataset,n_ways,n_shot,float(np.mean(acc_list))))