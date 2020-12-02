import numpy as np
import torch


def extract_sample(n_way, n_support, n_query, datax, datamap, datay, datasnt, n_unlabelled=0):
    sample, sample_map, sample_snt = [], [], []
    K = np.random.choice(np.unique(datay), n_way, replace=False)
    
    for cls in K:
        datax_cls = datax[datay == cls]
        datamap_cls = datamap[datay==cls]
        datasnt_cls = datasnt[datay==cls]

        p = np.random.permutation(len(datax_cls))

        perm_x = datax_cls[p]
        perm_map = datamap_cls[p]
        perm_snt = datasnt_cls[p]
            
        sample_cls = perm_x[:(n_support+n_query+n_unlabelled)]
        sample_map_cls = perm_map[:(n_support+n_query+n_unlabelled)]
        sample_snt_cls = perm_snt[:(n_support+n_query+n_unlabelled)]

        sample.append(sample_cls)
        sample_map.append(sample_map_cls)
        sample_snt.append(sample_snt_cls) 
        
    sample = np.array(sample)
    sample = torch.from_numpy(sample).int()
    
    sample_map = np.array(sample_map)
    sample_map = torch.from_numpy(sample_map).int()

    sample_snt = np.array(sample_snt)                            

    return({
          'sentence':sample_snt,
          'sequence': sample,
          'attention_mask': sample_map,
          'n_way': n_way,
          'n_support': n_support,
          'n_query': n_query,
          'n_unlabelled':n_unlabelled
          })