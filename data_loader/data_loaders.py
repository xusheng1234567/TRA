from torch.utils import data
from collections import namedtuple
import pandas as pd
import numpy as np
import os
import h5py
import logging
import random
import h5py
import os
import re
import logging
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
# import dgl
from torch.cuda.amp import autocast
import gc
import torch
from torch.utils.data import Dataset
import os
import numpy as np

class LoadDataset_from_numpy(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, np_dataset):
        super(LoadDataset_from_numpy, self).__init__()
        train_data=[]
        retrieval_train_data=[]
        h5_dataset=[]
        for np_file in np_dataset:
            temp_file=np_file.replace('npz','h5')
            h5_dataset.append(temp_file)
        for h5_file in h5_dataset:
            h5_file=[h5_file]
            X_train = np.load(np_dataset[0])["x"]
            data_generator=get_data_generator(h5_file, 
                    batch_size=X_train.shape[0], 
                    shuffle=True, 
                    feature_map=None, 
                    retrieval_configs={'used_cols': ['EEG-fpz-cz'], 'exact_match_cols': [], 'retrieval_pool_data': '/meida/cf/DISK/xs/AttnSleep_change_second/retrieval-k-means.h5', 'split_type': '10-fold', 'label_wise': False, 'pre_retrieval': True, 'enable_clean': False, 'qry_batch_size': 5000, 'db_chunk_size': 50000, 'device': 'cuda:0', 'topK': 5, 'used_col_indices': [0], 'exact_match_col_indices': None},
                    retrieval_pool_fname='self',
                    retrieval_augmented=True,
                    )
            for batch_data,batch_index in enumerate(data_generator):
                batch_index_retrieval=batch_index[5]
                batch_index=batch_index[1]
                train_data.append( batch_index)
                retrieval_train_data.append( batch_index_retrieval)
        topk=5
        train_data = np.concatenate(train_data, axis=0)
        retrieval_train_data = np.concatenate(retrieval_train_data, axis=0)
        train_data=np.array(train_data)
        retrieval_train_data=np.array(retrieval_train_data)
        X_train=(train_data[:, :, :3000])  
        y_train=(train_data[:, :, 3000:])
        X_train_retrieval=(retrieval_train_data[:, :, :3000])  
        y_train_retrieval=(retrieval_train_data[:, :, 3000:])     
        
        X_train_retrieval=X_train_retrieval.astype(np.float32)
        y_train_retrieval=y_train_retrieval.astype(np.float32)
        X_train=X_train.astype(np.float32)
        y_train=y_train.astype(np.float32)

        self.len = X_train.shape[0]
        self.retrieval_len=X_train_retrieval.shape[0]
        self.x_data = torch.from_numpy(X_train)
        self.x_data_retrieval=torch.from_numpy(X_train_retrieval)
        self.y_data = torch.from_numpy(y_train).long()
        self.y_data_retrieval = torch.from_numpy(y_train_retrieval).long()

        if len(self.x_data.shape) == 3:
            if self.x_data.shape[1] != 1:
                self.x_data = self.x_data.permute(0, 2, 1)
        else:
            self.x_data = self.x_data.unsqueeze(1)
        # Correcting the shape of input_retrieval to be (Batch_size, #channels, seq_len) where #channels=    
        if len(self.x_data_retrieval.shape) == 3:
            if self.x_data_retrieval.shape[1] != topk:
                self.x_data_retrieval = self.x_data_retrieval.permute(0, 2, 1)
        else:
            self.x_data_retrieval = self.x_data_retrieval.unsqueeze(1)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index],self.x_data_retrieval[index],self.y_data_retrieval[index]

    def __len__(self):
        return self.len
def data_generator_np(training_files, subject_files,adds_files, batch_size):
    train_dataset = LoadDataset_from_numpy(training_files)
    adds_dataset=LoadDataset_from_numpy(adds_files)
    test_dataset = LoadDataset_from_numpy(subject_files)
    adds_dataset.y_data=train_dataset.y_data.squeeze()
    train_dataset.y_data=train_dataset.y_data.squeeze()
    test_dataset.y_data=test_dataset.y_data.squeeze()
    # to calculate the ratio for the CAL
    all_ys = np.concatenate((train_dataset.y_data, test_dataset.y_data))
    all_ys = all_ys.tolist()
    num_classes = len(np.unique(all_ys))
    counts = [all_ys.count(i) for i in range(num_classes)]
    adds_loader=torch.utils.data.DataLoader(dataset=adds_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=False,
                                               num_workers=0)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=False,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0)
    return train_loader, test_loader,adds_loader, counts

class DataGenerator(data.DataLoader):
    def __init__(self, data_path, batch_size=32,
                 shuffle=False, num_workers=1,
                 feature_map=None,
                 graph_processor=None,
                 retrieval_configs=None,
                 retrieval_pool_fname=None,
                 retrieval_augmented=False,
                 **kwargs):
        if type(data_path) == list:
            data_path = data_path[0]
        data_array = load_hdf5(data_path)
        self.graph_processor = graph_processor
        if self.graph_processor:
            self.graph_processor = getattr(dataset_utils, graph_processor)
        retrieval_configs={'used_cols': ['EEG-fpz-cz'], 'exact_match_cols': [], 'retrieval_pool_data': '/media/cf/DISK/xs/AttnSleep_change_second/retrieval-k-means.h5', 'split_type': '10-fold', 'label_wise': False, 'pre_retrieval': True, 'enable_clean': False, 'qry_batch_size': 5000, 'db_chunk_size': 50000, 'device': 'cuda:0', 'topK':5, 'used_col_indices': [0], 'exact_match_col_indices': None}
        if retrieval_configs is not None:
            assert retrieval_configs["pre_retrieval"], "目前只实现了 pre-retrieval 策略"
            if retrieval_configs["pre_retrieval"]:
                if retrieval_pool_fname != "self":
                    logging.info(f"{retrieval_configs['split_type']} retrieval, pool file: {retrieval_pool_fname}")                
                    db_array = load_hdf5(retrieval_pool_fname)  
                data_root, data_fname = os.path.split(data_path)
                retrieval_save_path = os.path.join(
                    data_root,
                    f'retrieval_{retrieval_configs["topK"]}_' + data_fname)
                if os.path.exists(retrieval_save_path)and False:
                    retrieved_indices = load_hdf5(retrieval_save_path, "indices")  
                    retrieved_values = load_hdf5(retrieval_save_path, "values")  
                    retrieved_lens = load_hdf5(retrieval_save_path, "lens")  
                else:
                    if retrieval_pool_fname == 'self':
                        retrieval_data_array = data_array[:, retrieval_configs["used_col_indices"]].astype(int)
                        if retrieval_configs["label_wise"]:
                            retrieval_db_labels = data_array[:, -1].astype(int)
                        retrieved_indices = []
                        retrieved_values = []
                        retrieved_lens = []
                        fold_num = int(re.match("\d+-fold", retrieval_configs["split_type"]).group().split('-')[0])
                        fold_size = int(np.ceil(len(retrieval_data_array) / fold_num))
                        for fi in range(fold_num):
                            logging.info(f"{fold_num}-fold retrieval: 处理第 {fi} 个折")
                            fold_qry_data = retrieval_data_array[fi * fold_size: (fi + 1) * fold_size]  
                            fold_db_data = np.concatenate(
                                [retrieval_data_array[: fi * fold_size],
                                 retrieval_data_array[(fi + 1) * fold_size:]],
                                axis=0)  
                            fold_db_indices = np.concatenate(
                                [np.arange(fi * fold_size),
                                 np.arange((fi + 1) * fold_size, len(retrieval_data_array))],
                                axis=0)  
                            if retrieval_configs["label_wise"]:
                                fold_db_labels = np.concatenate(
                                    [retrieval_db_labels[: fi * fold_size],
                                     retrieval_db_labels[(fi + 1) * fold_size:]],
                                    axis=0)  
                                db_pos_indices = np.nonzero(fold_db_labels)[0]
                                fold_retrieved_pos_results = BM25_topk_retrieval(
                                    db_np_data=fold_db_data[db_pos_indices],
                                    qry_np_data=fold_qry_data, **retrieval_configs)
                                fold_retrieved_pos_indices = fold_db_indices[
                                    db_pos_indices[fold_retrieved_pos_results.indices]]  # BxK
                                fold_retrieved_pos_values = fold_retrieved_pos_results.values  # BxK
                                fold_retrieved_pos_lens = fold_retrieved_pos_results.lens  # B
                                db_neg_indices = np.nonzero(1 - fold_db_labels)[0]
                                fold_retrieved_neg_results = BM25_topk_retrieval(
                                    db_np_data=fold_db_data[db_neg_indices],
                                    qry_np_data=fold_qry_data, **retrieval_configs)
                                fold_retrieved_neg_indices = fold_db_indices[
                                    db_neg_indices[fold_retrieved_neg_results.indices]]  # BxK
                                fold_retrieved_neg_values = fold_retrieved_neg_results.values  # BxK
                                fold_retrieved_neg_lens = fold_retrieved_neg_results.lens  # B
                                retrieved_indices.append(
                                    np.concatenate([fold_retrieved_pos_indices, fold_retrieved_neg_indices],
                                                   axis=-1))  # Bx(2K)
                                retrieved_values.append(
                                    np.concatenate([fold_retrieved_pos_values, fold_retrieved_neg_values],
                                                   axis=-1))  # Bx(2K)
                                retrieved_lens.append(
                                    np.stack([fold_retrieved_pos_lens, fold_retrieved_neg_lens], axis=-1))  # Bx2
                            else:
                                fold_db_data=fold_db_data.reshape(-1,1)
                                fold_qry_data=fold_qry_data.reshape(-1,1)

                                fold_retrieved_results = BM25_topk_retrieval(db_np_data=fold_db_data,
                                                                             qry_np_data=fold_qry_data,
                                                                             **retrieval_configs)
                                retrieved_indices.append(fold_db_indices[fold_retrieved_results.indices])  # BxK
                                retrieved_values.append(fold_retrieved_results.values)  # BxK
                                retrieved_lens.append(fold_retrieved_results.lens)  # B
                        retrieved_indices = np.concatenate(retrieved_indices)  # QxK 或 Qx(2K)
                        retrieved_values = np.concatenate(retrieved_values)  # QxK 或 Qx(2K)
                        retrieved_lens = np.concatenate(retrieved_lens)  # Q 或 Qx2
                    else:
                        db_data = db_array[:, retrieval_configs["used_col_indices"]].astype(int)  # NdbxF'
                        qry_data = data_array[:, retrieval_configs["used_col_indices"]].astype(int)  # NqxF'
                        if retrieval_configs["label_wise"]:
                            db_labels = db_array[:, -1].astype(int)
                            db_pos_indices = np.nonzero(db_labels)[0]
                            retrieved_pos_results = BM25_topk_retrieval(
                                db_np_data=db_data[db_pos_indices],
                                qry_np_data=qry_data, **retrieval_configs)
                            retrieved_pos_indices = db_pos_indices[retrieved_pos_results.indices]  # QxK
                            retrieved_pos_values = retrieved_pos_results.values  # QxK
                            retrieved_pos_lens = retrieved_pos_results.lens  # Q
                            db_neg_indices = np.nonzero(1 - db_labels)[0]
                            retrieved_neg_results = BM25_topk_retrieval(
                                db_np_data=db_data[db_neg_indices],
                                qry_np_data=qry_data, **retrieval_configs)
                            retrieved_neg_indices = db_neg_indices[retrieved_neg_results.indices]  # QxK
                            retrieved_neg_values = retrieved_neg_results.values  # QxK
                            retrieved_neg_lens = retrieved_neg_results.lens  # Q
                            retrieved_indices = np.concatenate([retrieved_pos_indices, retrieved_neg_indices],
                                                               axis=-1)  # Qx(2K)
                            retrieved_values = np.concatenate([retrieved_pos_values, retrieved_neg_values],
                                                              axis=-1)  # Qx(2K)
                            retrieved_lens = np.stack([retrieved_pos_lens, retrieved_neg_lens], axis=-1)  # Qx2
                        else:
                            retrieved_results = BM25_topk_retrieval(db_np_data=db_data,
                                                                    qry_np_data=qry_data,
                                                                    **retrieval_configs)
                            retrieved_indices = retrieved_results.indices
                            retrieved_values = retrieved_results.values
                            retrieved_lens = retrieved_results.lens
                    save_hdf5(retrieved_indices, retrieval_save_path, "indices")  # QxK 或 Qx(2K)
                    save_hdf5(retrieved_values, retrieval_save_path, "values")  # QxK 或 Qx(2K)
                    save_hdf5(retrieved_lens, retrieval_save_path, "lens")  # Q 或 Qx2
                if retrieval_augmented:
                    self.dataset = Dataset(
                        darray=data_array,
                        feature_map=feature_map,
                        graph_processor=self.graph_processor,
                        retr_pool_darray=data_array if retrieval_pool_fname == 'self' else db_array,  
                        retr_indices=retrieved_indices,  
                        retr_values=retrieved_values,  
                        retr_lens=retrieved_lens)  
                else:
                    logging.info("[[WARNING]] dataloader 提供了检索结果，但模型未启用 retrieval-augmentated 模式。")
                    self.dataset = Dataset(
                        darray=data_array[:, 0],  
                        feature_map=feature_map,
                        graph_processor=self.graph_processor)
            else:
                raise NotImplementedError("我们仅实现了 pre-retrieval 策略")
        else:
            assert not retrieval_augmented, "retrieval-augmented 模式需要数据格式为 [Bx(1+K)x(F+1)]"
            self.dataset = Dataset(
                darray=data_array,
                feature_map=feature_map,
                graph_processor=self.graph_processor)

        super(DataGenerator, self).__init__(dataset=self.dataset, batch_size=batch_size,
                                            shuffle=shuffle, num_workers=num_workers,
                                            collate_fn=graph_collate_fn if self.graph_processor else None)
        self.num_blocks = 1  
        self.num_batches = int(np.ceil(len(self.dataset) * 1.0 / self.batch_size))  
        self.num_samples = len(data_array) 
        if data_array.ndim == 2:
            self.num_positives = data_array[:, -1].sum()  
        elif data_array.ndim == 3:
            self.num_positives = data_array[:, 0, -1].sum()  
        else:
            raise RuntimeError("data_array 必须为 [Nx(F+1)] 或 [Nx(K+1)x(F+1)] 的格式")

        self.num_negatives = self.num_samples - self.num_positives  

    def __len__(self):
        """
        返回批次数量。
        """
        return self.num_batches
class Dataset(data.Dataset):
    def __init__(self, darray,
                 feature_map=None,
                 graph_processor=None,
                 retr_pool_darray=None,
                 retr_indices=None,
                 retr_values=None,
                 retr_lens=None):
        self.darray = darray # QxF
        self.graph_processor = graph_processor
        self.retr_pool_darray = None
        self.retr_indices = None
        self.retr_values = None
        self.retr_lens = None
        self.retrieval_augmented = False
        if retr_pool_darray is not None and retr_indices is not None and \
            retr_values is not None and retr_lens is not None:
            self.retr_pool_darray = retr_pool_darray  # QxF (pool=='self') or NdbxF (pool!='self')
            self.retr_indices = retr_indices  # QxK or Qx(2K)
            self.retr_values = retr_values  # QxK or Qx(2K)
            self.retr_lens = retr_lens  # Q or Qx2
            self.retrieval_augmented = True
            assert len(self.darray) == len(self.retr_indices) == \
                len(self.retr_values) == len(self.retr_lens), \
                f"darray.len = {len(self.darray)}, retr_indices.len = {len(self.retr_indices)}, retr_values.len = {len(self.retr_values)}, retr_lens.len = {len(self.retr_lens.shape)}"
            assert self.retr_indices.shape[-1] == self.retr_values.shape[-1]
        if self.graph_processor:
            self.darray = self.graph_processor.convert_indices(
                self.darray, feature_map.feature_specs)
            if self.retrieval_augmented and id(self.darray) != id(self.retr_pool_darray): # avoid repeated conversion
                self.retr_pool_darray = self.graph_processor.convert_indices(
                    self.retr_pool_darray, feature_map.feature_specs)
    def __len__(self):
        return len(self.darray)
    def __getitem__(self, index):
            darray_i = self.darray[index] # (F+1)
            retrieved_darray_i = self.retr_pool_darray[self.retr_indices[index]] # Kx(F+1) or (2K)x(F+1)
            darray_i = np.expand_dims(darray_i, 0)  # 1x(F+1)
                
            X_i = darray_i[..., :-1] # F or (1+K)xF or (1+2K)xF
            y_i = darray_i[..., -1] # () or (1+K) or (1+2K)
            retrieval_x_i=retrieved_darray_i[..., :-1]
            retriebal_y_i = retrieved_darray_i[..., -1]

            if self.graph_processor:
                X_i = self.graph_processor.build_instance_graph(X_i, y_i)
            if self.retrieval_augmented:
                return X_i, y_i, self.retr_values[index], self.retr_lens[index],retrieval_x_i,retriebal_y_i # the last two: (K) or (2K), () or (2)
            return X_i, y_i  # F, ()
def load_hdf5(data_path, key=None, verbose=True):
    if verbose:
        logging.info('Loading data from h5: ' + data_path)
    with h5py.File(data_path, 'r') as hf:
        if key is not None:
            data_array = hf[key][()]
        else:
            data_array = hf[list(hf.keys())[0]][()]

    return data_array
def BM25_topk_retrieval(
    db_np_data: np.ndarray,  
    qry_np_data: np.ndarray,  
    exact_match_col_indices: list = None,  
    qry_batch_size: int = None,  
    db_chunk_size: int = None,  
    device: str = 'cpu',  
    topK: int = 10,  
    enable_clean: bool = False,  
    **kwargs  
):
    ResultsNamedTuple = namedtuple("ResultsNameTuple", ["values", "indices", "lens"])  

    def sort_results(values, indices):
        # values: BxK, indices: BxK
        drop_mask = (values == 0)  
        indices[drop_mask] = -1  
        results = torch.sort(values, descending=True)  
        values = results.values  
        indices = torch.gather(indices, -1, results.indices)  
        lens = drop_mask.shape[-1] - drop_mask.sum(-1)  
        if enable_clean:
            del drop_mask  
            gc.collect()  
        return ResultsNamedTuple(values, indices, lens)  

    def padded_topk(input_values, K, index_offs=None):
        assert input_values.ndim == 2, "input shape must be [BxN]" 
        output_lens = torch.zeros_like(input_values[:, 0], dtype=int)  
        if K >= input_values.shape[-1]:
            output_values = F.pad(input_values, (0, K - input_values.shape[-1]))  
            output_indices = torch.zeros_like(output_values, dtype=torch.long)  
            for col_i in range(input_values.shape[-1]):
                output_indices[:, col_i] = col_i  
            if index_offs:
                output_indices += index_offs  
            output_indices[:, input_values.shape[-1]:] = -1  
            output_lens[:] = input_values.shape[-1]  
        else:
            output_results = torch.topk(input_values, k=K)  
            output_values = output_results.values
            output_indices = output_results.indices
            if index_offs:
                output_indices += index_offs  
            output_lens[:] = K  
        return ResultsNamedTuple(output_values, output_indices, output_lens)  

    def masked_gather(input, index, mask_index_value=-1):
        if mask_index_value not in index:
            return torch.gather(input, -1, index)  
        else:
            mask = (index == mask_index_value)  
            index[mask] = 0  
            results = torch.gather(input, -1, index)  
            results[mask] = mask_index_value  
            if enable_clean:
                del mask  
                gc.collect()  
            return results  

    def masked_indexing(input, index, mask_index_value=-1):
        if mask_index_value not in index:
            return input[index]  
        else:
            mask = (index == mask_index_value)  
            index[mask] = 0  
            results = input[index]  
            results[mask] = mask_index_value  
            if enable_clean:
                del mask  
                gc.collect()  
            return results  

    def map_data_to_IDF_v1(np_data, IDF_stats):
        IDF_np_data = np.zeros_like(np_data, dtype=float)  
        for i, col_IDF_stats in enumerate(IDF_stats):
            IDF_np_data[:, i] = np.vectorize(lambda x: col_IDF_stats.get(x, 0))(np_data[:, i])  
        return IDF_np_data  
    def map_data_to_IDF_v2(np_data, IDF_stats):
        # NOT safe: need to assure that the query and the db have the same value set, or will cause IndexError
        IDF_np_data = []  
        for i, col_IDF_stats in enumerate(IDF_stats):
            IDF_np_data.append(col_IDF_stats.values[map_indices(col_IDF_stats.index.to_numpy(), np_data[:, i], missing=-1)])
        return np.stack(IDF_np_data, axis=-1)  
    map_data_to_IDF = map_data_to_IDF_v1  

    if exact_match_col_indices: 
        db_df = pd.DataFrame(db_np_data)  
        db_groups = pd.Series(db_df.groupby(exact_match_col_indices).groups)  
        exm_cols_mask = np.zeros(db_np_data.shape[-1], dtype=bool)  
        exm_cols_mask[exact_match_col_indices] = True  
        rest_cols_mask = ~exm_cols_mask  
        qry_exm_cols_df = pd.DataFrame(qry_np_data[:, exm_cols_mask])  
        qry_exm_cols_df = qry_exm_cols_df.set_index(list(qry_exm_cols_df.columns))  
        qry_exm_grp_ids = db_groups.index.get_indexer(qry_exm_cols_df.index)  
        db_np_data = db_np_data[:, rest_cols_mask]  
        qry_np_data = qry_np_data[:, rest_cols_mask]  
        if enable_clean:
            del exm_cols_mask, rest_cols_mask, qry_exm_cols_df
            gc.collect()  
    N = len(db_np_data)  
    db_df = pd.DataFrame(db_np_data)  
    IDF_stats = []  
    for col in db_df:
        col_IDF_stats = db_df[col].value_counts()  
        col_IDF_stats = np.log(N / col_IDF_stats)  
        IDF_stats.append(col_IDF_stats)  

    if db_chunk_size is None:  
        db_data = torch.from_numpy(db_np_data).to(device)  

    qry_batch_size = len(qry_np_data) if qry_batch_size is None else qry_batch_size  
    topK_values = np.zeros((len(qry_np_data), topK), dtype=float)  
    topK_indices = np.full((len(qry_np_data), topK), -1, dtype=int)  
    topK_indices_len = np.zeros(len(qry_np_data), dtype=int)  

    for qry_idx in tqdm(range(0, len(qry_np_data), qry_batch_size), desc="retrieve samples"):  
        if exact_match_col_indices:  
            qry_exm_grp_ids_batch = qry_exm_grp_ids[qry_idx: qry_idx + qry_batch_size]  
            valid_qry_exm_grp_ids_batch = qry_exm_grp_ids_batch[qry_exm_grp_ids_batch != -1]  
            if len(valid_qry_exm_grp_ids_batch) == 0:
                continue  
            exm_indices_batch = db_groups[db_groups.index[valid_qry_exm_grp_ids_batch]]  
            exm_indices_batch = pad_sequences(exm_indices_batch, padding='post',
                                              maxlen=topK if qry_np_data.shape[-1] == 0 else None,
                                              value=-1, dtype="int64")  
            exm_max_size_batch = exm_indices_batch.shape[-1]  
            if enable_clean:
                del valid_qry_exm_grp_ids_batch
                gc.collect()  

        if exact_match_col_indices and exm_max_size_batch <= topK:
            # padding to topK length
            topK_indices_len_batch = (exm_indices_batch != -1).sum(-1)  
            topK_indices_batch = np.pad(exm_indices_batch,
                                        ((0, 0), (0, topK - exm_max_size_batch)),
                                        constant_values=-1)  
            topK_values_batch = (topK_indices_batch != -1).astype(float)  
            if enable_clean:
                del exm_indices_batch
                gc.collect()

        elif qry_np_data.shape[-1] > 0:  
            qry_data_batch = qry_np_data[qry_idx: qry_idx + qry_batch_size]  
            if exact_match_col_indices:
                # filter out those query samples without any matched retrieval sample
                qry_data_batch = qry_data_batch[qry_exm_grp_ids_batch != -1]  
            qry_IDF_data_batch = map_data_to_IDF(qry_data_batch, IDF_stats)  
            qry_data_batch = torch.from_numpy(qry_data_batch).to(device)  
            qry_IDF_data_batch = torch.from_numpy(qry_IDF_data_batch).to(device)  

            # aggregate the exm ids of samples in current batch as the batch-wise db
            if exact_match_col_indices:
                exm_indices_batch = torch.from_numpy(exm_indices_batch).long().to(device) 
                all_exm_indices_batch = torch.unique(exm_indices_batch)  
                if all_exm_indices_batch[0] == -1:
                    all_exm_indices_batch = all_exm_indices_batch[1:]  
                mapped_exm_indices_batch = map_indices(all_exm_indices_batch, exm_indices_batch, missing=-1, is_key_sorted=True)  

            if db_chunk_size is None:  
                if exact_match_col_indices:
                    exm_values_batch = (mapped_exm_indices_batch != -1).float()  
                    db_data_batch = db_data[all_exm_indices_batch]  
                    BM25_values_batch = ((qry_data_batch.unsqueeze(1) == db_data_batch.unsqueeze(0)) * qry_IDF_data_batch.unsqueeze(1)).sum(-1)  
                    BM25_values_batch = masked_gather(BM25_values_batch, mapped_exm_indices_batch)  
                    BM25_values_batch = (BM25_values_batch + 1) * exm_values_batch  
                    if enable_clean:
                        del exm_values_batch, mapped_exm_indices_batch, db_data_batch
                        gc.collect()  
                else:
                    BM25_values_batch = ((qry_data_batch.unsqueeze(1) == db_data.unsqueeze(0)) * qry_IDF_data_batch.unsqueeze(1)).sum(-1)  
                if enable_clean:
                    del qry_data_batch, qry_IDF_data_batch
                    gc.collect()
                topK_results_batch = padded_topk(BM25_values_batch, topK)  
                topK_values_batch = topK_results_batch.values  
                topK_indices_batch = topK_results_batch.indices  
                if enable_clean:
                    del BM25_values_batch
                    gc.collect()
                # gather the global indices according to the indices in the filtered list 'exm_indices_batch' (shape: BxExF)
                if exact_match_col_indices:
                    topK_indices_batch = masked_gather(exm_indices_batch, index=topK_indices_batch)  
                    if enable_clean:
                        del exm_indices_batch
                        gc.collect()
                topK_results_batch = sort_results(topK_values_batch, topK_indices_batch)  
                topK_values_batch = topK_results_batch.values.cpu().numpy()  
                topK_indices_batch = topK_results_batch.indices.cpu().numpy()  
                topK_indices_len_batch = topK_results_batch.lens.cpu().numpy()  
            else:  
                local_topK_values_batch = []
                local_topK_indices_batch = []
                if exact_match_col_indices:
                    db_np_data_batch = db_np_data[all_exm_indices_batch.cpu().numpy()] 
                    mapped_exm_indices_batch = mapped_exm_indices_batch.cpu().numpy()  
                    exm_values_batch = np.zeros((len(mapped_exm_indices_batch),
                                                 len(all_exm_indices_batch) + 1),
                                                dtype=float)  
                    exm_values_invalid_mask_batch = (mapped_exm_indices_batch == -1)  
                    mapped_exm_indices_batch[exm_values_invalid_mask_batch] = len(all_exm_indices_batch)  
                    np.put_along_axis(arr=exm_values_batch,  
                                      indices=mapped_exm_indices_batch,
                                      values=(~exm_values_invalid_mask_batch),
                                      axis=-1)
                    exm_values_batch = exm_values_batch[:, :len(all_exm_indices_batch)]  
                    if enable_clean:
                        del mapped_exm_indices_batch, exm_values_invalid_mask_batch
                        gc.collect()

                    for db_idx in range(0, len(db_np_data_batch), db_chunk_size):
                        local_db_data = torch.from_numpy(db_np_data_batch[db_idx: db_idx + db_chunk_size]).to(device)  
                        local_exm_values_batch = torch.from_numpy(exm_values_batch[:, db_idx: db_idx + db_chunk_size]).to(device)  
                        # B: qry_batch_size, C: database_chunk_size, F: field_num
                        # (Bx1xF compare 1xCxF) * Bx1xF => BxCxF => BxC
                        local_BM25_values_batch = ((qry_data_batch.unsqueeze(1) == local_db_data.unsqueeze(0)) * qry_IDF_data_batch.unsqueeze(1)).sum(-1)
                        local_BM25_values_batch = (local_BM25_values_batch + 1) * local_exm_values_batch  
                        if enable_clean:
                            del local_db_data, local_exm_values_batch
                            gc.collect()  
                        local_results = padded_topk(local_BM25_values_batch, topK, db_idx)  
                        local_topK_values_batch.append(local_results.values)  
                        local_topK_indices_batch.append(local_results.indices)  
                        if enable_clean:
                            del local_BM25_values_batch
                            gc.collect()
                else:
                    for db_idx in range(0, len(db_np_data), db_chunk_size):
                        local_db_data = torch.from_numpy(db_np_data[db_idx: db_idx + db_chunk_size]).to(device)  
                        # B: qry_batch_size, C: database_chunk_size, F: field_num
                        # (Bx1xF compare 1xNxF) * Bx1xF => BxCxF => BxC
                        local_BM25_values_batch = ((qry_data_batch.unsqueeze(1) == local_db_data.unsqueeze(0)) * qry_IDF_data_batch.unsqueeze(1)).sum(-1)
                        local_results = padded_topk(local_BM25_values_batch, topK, db_idx)  
                        local_topK_values_batch.append(local_results.values)  
                        local_topK_indices_batch.append(local_results.indices)  
                        if enable_clean:
                            del local_db_data, local_BM25_values_batch
                            gc.collect()  
                local_topK_values_batch = torch.cat(local_topK_values_batch, dim=-1)
                local_topK_indices_batch = torch.cat(local_topK_indices_batch, dim=-1)
                topK_results_batch = padded_topk(local_topK_values_batch, topK)
                topK_values_batch = topK_results_batch.values
                topK_indices_batch = masked_gather(local_topK_indices_batch, index=topK_results_batch.indices)  
                if enable_clean:
                    del local_topK_values_batch, local_topK_indices_batch
                    gc.collect()
                if exact_match_col_indices:
                    topK_indices_batch = masked_indexing(all_exm_indices_batch, topK_indices_batch)  
                    if enable_clean:
                        del all_exm_indices_batch
                        gc.collect()

                topK_results_batch = sort_results(topK_values_batch, topK_indices_batch)  
                topK_values_batch = topK_results_batch.values.cpu().numpy()  
                topK_indices_batch = topK_results_batch.indices.cpu().numpy()  
                topK_indices_len_batch = topK_results_batch.lens.cpu().numpy()  

        else:  
            assert exact_match_col_indices is not None, "detected empty query tensor input"
            topK_indices_batch = exm_indices_batch
            topK_indices_len_batch = (topK_indices_batch != -1).sum(-1)  
            topK_values_batch = (topK_indices_batch != -1).astype(float)  
        if exact_match_col_indices:
            topK_values[qry_idx: qry_idx + qry_batch_size][qry_exm_grp_ids_batch != -1] = topK_values_batch
            topK_indices[qry_idx: qry_idx + qry_batch_size][qry_exm_grp_ids_batch != -1] = topK_indices_batch
            topK_indices_len[qry_idx: qry_idx + qry_batch_size][qry_exm_grp_ids_batch != -1] = topK_indices_len_batch
            if enable_clean:
                del qry_exm_grp_ids_batch
                gc.collect()  
        else:
            topK_values[qry_idx: qry_idx + qry_batch_size] = topK_values_batch
            topK_indices[qry_idx: qry_idx + qry_batch_size] = topK_indices_batch
            topK_indices_len[qry_idx: qry_idx + qry_batch_size] = topK_indices_len_batch
        if enable_clean:
            del topK_values_batch, topK_indices_batch, topK_indices_len_batch
            gc.collect()  

    if enable_clean:
        del IDF_stats
        gc.collect()  

    return ResultsNamedTuple(topK_values, topK_indices, topK_indices_len)  

def map_indices(keys, queries, missing=-1, is_key_sorted=False, enable_clean=False):
    assert keys.ndim == 1
    if isinstance(keys, np.ndarray) and isinstance(queries, np.ndarray):
        if is_key_sorted:
            sorter = np.arange(len(keys))
        else:
            sorter = np.argsort(keys, kind='mergesort')
        insertion = np.searchsorted(keys, queries, sorter=sorter)
    elif isinstance(keys, torch.Tensor) and isinstance(queries, torch.Tensor):
        if is_key_sorted:
            sorter = torch.arange(len(keys), device=keys.device)
        else:
            sorter = torch.argsort(keys, stable=True)
        insertion = torch.searchsorted(keys, queries, sorter=sorter)
    else:
        raise TypeError(f"The type of 'keys' ({type(keys)}) doesn't match the 'queries' ({type(queries)})")

    indices = sorter[insertion]
    invalid = keys[indices] != queries
    indices[invalid] = missing
    if enable_clean:
        del sorter, insertion, invalid
        gc.collect()
    return indices

def graph_collate_fn(batch):
    if len(batch[0]) == 2:
        batch_graphs, batch_labels = map(list, zip(*batch))
        batch_graphs = dgl.batch(batch_graphs)
        batch_labels = torch.from_numpy(np.stack(batch_labels))
        return batch_graphs, batch_labels
    elif len(batch[0]) == 4:
        batch_graphs, batch_labels, batch_retr_values, batch_retr_lens = map(list, zip(*batch))
        batch_graphs = dgl.batch(batch_graphs)
        batch_labels = torch.from_numpy(np.stack(batch_labels))
        batch_retr_values = torch.from_numpy(np.stack(batch_retr_values))
        batch_retr_lens = torch.from_numpy(np.stack(batch_retr_lens))
        return batch_graphs, batch_labels, batch_retr_values, batch_retr_lens


def save_hdf5(data_array, data_path, key="data"):
    logging.info("Saving data to h5: " + data_path)
    dir_name = os.path.dirname(data_path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with h5py.File(data_path, 'w') as hf:
        hf.create_dataset(key, data=data_array)


def get_data_generator(
    data_path_list,
    batch_size=32,
    shuffle=False,
    num_workers=0,
    feature_map=None,
    retrieval_configs=None,
    retrieval_pool_fname=None,
    retrieval_augmented=False,
    **kwargs):

    assert len(data_path_list) > 0, "invalid data files or paths."
    if len(data_path_list) == 1:
        return DataGenerator(data_path=data_path_list[0],
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             feature_map=feature_map,
                             retrieval_configs=retrieval_configs,
                             retrieval_pool_fname=retrieval_pool_fname,
                             retrieval_augmented=retrieval_augmented,
                             **kwargs)