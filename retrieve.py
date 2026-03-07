import torch
import torch.nn.functional as F
import numpy as np
from data_generation.encoders import get_encoders




class Retriever():
    def __init__(
        self,
        retrieve_key='DINO', #['DINO', 'CLIP']
        state_num=5,
        metric='cosine',
        traj_metric = 'ot', #'ot','sdtw'
        re_history_len=5,
        retrieve_len=5,
        device='cuda'

    ):
        super(Retriever, self).__init__()

        self.state_num = state_num
        self.metric = metric
        assert metric in ['cosine','l2']

        self.traj_metric = traj_metric
        assert traj_metric in ['ot','sdtw','none']

        self.retrieve_len = retrieve_len
        self.re_history_len = re_history_len
        self.retrieve_key = retrieve_key
        self.encoder = None
        self.device = device


    def get_state_subset(self,current_traj,retrieve_list,ex_id=(None,None)):
        env_id,traj_id = ex_id
        if env_id is not None:
            current_state = current_traj[-1][self.retrieve_key]
        else:
            #current_img = current_traj[-1].transpose(1, 2, 0)
            current_state =current_traj[-1]

        # Flatten dataset into [N, D] tensor and track indices
        all_states = []
        indices = []
        if isinstance(current_state, np.ndarray):
            current_state = torch.from_numpy(current_state)
        if current_state.ndim == 1:
            current_state = current_state.unsqueeze(0)

        for task_idx in retrieve_list.keys():
            task_demo = retrieve_list[task_idx]
            for traj_idx, traj in enumerate(task_demo):
                if (traj_idx == traj_id) & (task_idx == env_id):
                    continue
                re_traj = traj["observation"]['retrieve_feature']
                # need to change
                for state_idx, state in enumerate(re_traj):
                    state = torch.from_numpy(state[self.retrieve_key])
                    all_states.append(state.unsqueeze(0))  # Shape: [1, D]
                    indices.append((task_idx,traj_idx, state_idx))

        all_states_tensor = torch.cat(all_states, dim=0)  # Shape: [N, D]

        #for query_state in current_state:  # Iterate over batch
        if self.metric == 'cosine':
            # Cosine similarity: higher is better
            sim_scores = F.cosine_similarity(current_state, all_states_tensor, dim=1)
        elif self.metric == 'l2':
            # L2 distance: negative norm, higher is better
            sim_scores = -torch.norm(all_states_tensor - current_state, p=2, dim=1)

        elif self.metric == None:
            # No state filter
            sim_scores = None * torch.ones_like(all_states_tensor)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

        result = list(zip(indices, sim_scores.tolist()))
        if self.metric is not None:
            result.sort(key=lambda x: x[1], reverse=True)

        state_subset =  result[:self.state_num]

        return state_subset

    def get_state_subset_from_task(self,current_traj,retrieve_list,ex_id=None,env_id=None):
        if ex_id is not None:
            current_state = current_traj[-1][self.retrieve_key]
        else:
            #current_img = current_traj[-1].transpose(1, 2, 0)
            current_state =current_traj[-1]

        # Flatten dataset into [N, D] tensor and track indices
        all_states = []
        indices = []
        if isinstance(current_state, np.ndarray):
            current_state = torch.from_numpy(current_state)
        if current_state.ndim == 1:
            current_state = current_state.unsqueeze(0)

        for traj_idx, traj in enumerate(retrieve_list):
            if (traj_idx == ex_id) :
                continue
            if 'observation' in traj.keys():
                traj = traj['observation']
            re_traj = traj['retrieve_feature']
            # need to change
            for state_idx, state in enumerate(re_traj):
                state = torch.from_numpy(state[self.retrieve_key])
                all_states.append(state.unsqueeze(0))  # Shape: [1, D]
                indices.append((traj_idx, state_idx))

        all_states_tensor = torch.cat(all_states, dim=0)  # Shape: [N, D]

        #for query_state in current_state:  # Iterate over batch
        if self.metric == 'cosine':
            # Cosine similarity: higher is better
            sim_scores = F.cosine_similarity(current_state, all_states_tensor, dim=1)
        elif self.metric == 'l2':
            # L2 distance: negative norm, higher is better
            sim_scores = -torch.norm(all_states_tensor - current_state, p=2, dim=1)

        elif self.metric == None:
            # No state filter
            sim_scores = None * torch.ones_like(all_states_tensor)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

        result = list(zip(indices, sim_scores.tolist()))
        if self.metric is not None:
            result.sort(key=lambda x: x[1], reverse=True)
            #results.append(result[:self.state_num])

        state_subset =  result[:self.state_num]

        return state_subset

    def get_traj_index_from_subset(self,current_traj,state_subset,retrieve_list):

        length = min( len(current_traj),self.re_history_len )
        current_history_traj = current_traj[-length:]
        best_cost = float('inf')

        candidate_list = []
        index_map = []

        for i, ((task_id, traj_idx, state_idx), score) in enumerate(state_subset):
            if self.traj_metric == 'ot':
                expert_start = max(0,state_idx-length+1)
                expert_history_traj = retrieve_list[task_id][traj_idx]['retrieve_feature'][expert_start:state_idx+1]
                metrics = self.temporal_ot_metrics(expert_history_traj, current_history_traj, mask_k=1, epsilon=0.01, niter=100)
                if metrics["transport_cost"] < best_cost:
                    best_cost = metrics["transport_cost"]
                    best_index = task_id,traj_idx, state_idx

            elif self.traj_metric =='sdtw':
                expert_history_traj=  retrieve_list[task_id][traj_idx]["observation"]['retrieve_feature'][0:state_idx+1]

            candidate_list.append(expert_history_traj)
            index_map.append((task_id, traj_idx, state_idx))

        best_idx, best_dist, start_idx, end_idx,path_len = self.batch_sdtw_metrics(
            current_history_traj, candidate_list, device=self.device
        )

        # Map back to original indices
        best_task_id, best_traj_idx, best_state_idx = index_map[best_idx]

        return (best_task_id,best_traj_idx, start_idx,end_idx,best_dist)

    def get_traj_index_from_subset_traj(self,current_traj,state_subset,retrieve_list):

        length = min( len(current_traj),self.re_history_len )
        current_history_traj = current_traj[-length:]
        best_cost = float('inf')

        candidate_list = []
        index_map = []

        for i, ((traj_idx, state_idx), score) in enumerate(state_subset):
            traj = retrieve_list[traj_idx]
            if 'observation' in traj.keys():
                traj = traj['observation']
            if self.traj_metric == 'ot':
                expert_start = max(0,state_idx-length+1)
                expert_history_traj = traj['retrieve_feature'][expert_start:state_idx+1]
                metrics = self.temporal_ot_metrics(expert_history_traj, current_history_traj, mask_k=1, epsilon=0.01, niter=100)
                if metrics["transport_cost"] < best_cost:
                    best_cost = metrics["transport_cost"]
                    best_index = traj_idx, state_idx

            elif self.traj_metric =='sdtw':
                expert_history_traj=  traj['retrieve_feature'][0:state_idx+1]

            candidate_list.append(expert_history_traj)
            index_map.append(( traj_idx, state_idx))

        best_idx, best_dist, start_idx, end_idx,path_len = self.batch_sdtw_metrics(
            current_history_traj, candidate_list, device=self.device
        )

        # Map back to original indices
        best_traj_idx, best_state_idx = index_map[best_idx]
        #print(f'Get the {best_idx}th in state subset')
        return (best_traj_idx, start_idx,end_idx,best_dist,path_len)

    def state_encode(self,img):
        if self.encoder is None:
            encoder = get_encoders([self.retrieve_key])
            self.encoder = encoder[self.retrieve_key]
        current_img = img.transpose(1, 2, 0)
        current_state = self.encoder.encode(current_img)

        return current_state

    def temporal_ot_metrics(self,traj1, traj2, mask_k=3, epsilon=0.01, niter=100):
        """
        Compute temporal Optimal Transport metrics between two trajectories in PyTorch.

        Args:
            traj1, traj2: torch.Tensor of shape [T, D] - encoded trajectories
            mask: torch.Tensor of shape [T, T] - binary mask (1: allow, 0: block)
            epsilon: float - entropy regularization
            niter: int - number of Sinkhorn iterations

        Returns:
            dict with transport_cost, diagonal_mass, entropy, avg_step_shift
        """
        T1, T2 = traj1.shape[0], traj2.shape[0]
        mask = temporal_mask(T1, T2, k=mask_k)

        # Uniform marginals
        traj1_pot = np.ones(traj1.shape[0]) / traj1.shape[0]
        traj2_pot = np.ones(traj2.shape[0]) / traj2.shape[0]

        # Cost matrix (cosine distance)
        traj1_feature = np.array([state[self.retrieve_key] for state in traj1])
        traj2_feature = np.array([state[self.retrieve_key] for state in traj2])
        cost_matrix = cosine_distance_matrix(torch.from_numpy(traj1_feature),
                                             torch.from_numpy(traj2_feature)
                                             )  # [T1, T2]
        c_m = cost_matrix.data.detach().cpu().numpy()

        transport_plan = mask_sinkhorn(traj1_pot,
                                       traj2_pot,
                                       c_m,
                                       mask,
                                       epsilon,
                                       numItermax=niter)

        transport_plan = torch.from_numpy(transport_plan)

        # Metrics
        transport_cost = torch.sum(transport_plan * c_m).item()
        #diagonal_mass = torch.sum(torch.diag(transport_plan)).item()

        return transport_cost

    def sdtw_metrics(self,query_dict, sequence_dict,device):
        """
        Soft-DTW style alignment between two sequences.
        query: [Tq, D]
        sequence: [Ts, D]
        """
        query = np.array([state[self.retrieve_key] for state in query_dict])
        sequence = np.array([state[self.retrieve_key] for state in sequence_dict])
        query = torch.from_numpy(query).to(device)
        sequence = torch.from_numpy(sequence).to(device)

        m = query.shape[0]  # len(query)
        n = sequence.shape[0]
        # Compute distance matrix [m, n]
        # dist_matrix[i, j] = ||query[i] - sequence[j]||^2
        diff = query[:, None, :] - sequence[None, :, :]  # [m, n, D]
        dist_matrix = torch.sum(diff ** 2, dim=2)  # [m, n]

        D = torch.full((m + 1, n + 1), float('inf'), device=device)
        D[0, :] = 0  # allow starting anywhere

        for i in range(1, m + 1):
            D[i, 1:] = dist_matrix[i - 1] + torch.min(
                torch.stack([D[i - 1, :-1], D[i - 1, 1:], D[i, :-1]], dim=0), dim=0
            ).values

        end_idx = torch.argmin(D[m, 1:]).item()
        best_dist = D[m, end_idx + 1].item()

        # Backtrack to find alignment start
        i, j = m, end_idx + 1
        while i > 0:
            cost = dist_matrix[i - 1, j - 1]
            if D[i, j] == cost + D[i - 1, j - 1]:
                i -= 1
                j -= 1
            elif D[i, j] == cost + D[i - 1, j]:
                i -= 1
            else:
                j -= 1
        start_idx = j
        # slice the matched subsequence
        #subseq = sequence[start_idx:end_idx + 1]

        return best_dist, start_idx, end_idx


    def batch_sdtw_metrics(self,query_state, candidate_list, device):
        """
        Compute SDTW between a query sequence and multiple candidate sequences in parallel.
        Handles candidates of different lengths using padding + masking.

        Returns:
            best_idx: index of most similar candidate
            best_dist: SDTW distance of best candidate
            start_idx, end_idx: aligned subsequence indices in the best candidate
        """
        num_candidates = len(candidate_list)
        if isinstance(query_state[0],dict):
            query_state = np.array([s[self.retrieve_key] for s in query_state])
        # Convert query to tensor
        query = torch.from_numpy(np.array(query_state)).to(device)
        query = F.normalize(query, p=2, dim=-1)
        Tq, D = query.shape

        # Find max candidate length for padding
        Ts_max = max(len(c) for c in candidate_list)

        # Prepare candidate tensor and mask
        candidates_tensor = torch.zeros((num_candidates, Ts_max, D), device=device)
        mask = torch.zeros((num_candidates, Ts_max), dtype=torch.bool, device=device)

        for i, c in enumerate(candidate_list):
            if isinstance(c[0],dict):
                c = [s[self.retrieve_key] for s in c]
            seq = torch.from_numpy(np.array(c))

            candidates_tensor[i, :seq.shape[0]] = F.normalize(seq, p=2, dim=-1)
            mask[i, :seq.shape[0]] = 1  # valid positions

        # Distance matrices [B, Tq, Ts_max]
        query_exp = query.unsqueeze(0).expand(num_candidates, -1, -1)
        diff = query_exp[:, :, None, :] - candidates_tensor[:, None, :, :]
        dist_matrix = torch.sum(diff ** 2, dim=-1)

        # Initialize DP table
        B = num_candidates
        D_table = torch.full((B, Tq + 1, Ts_max + 1), float('inf'), device=device)
        D_table[:, 0, :] = 0  # allow starting anywhere

        # DP loop
        for i in range(1, Tq + 1):
            prev1 = D_table[:, i - 1, :-1]
            prev2 = D_table[:, i - 1, 1:]
            prev3 = D_table[:, i, :-1]
            D_table[:, i, 1:] = dist_matrix[:, i - 1] + torch.min(torch.stack([prev1, prev2, prev3], dim=0),
                                                                  dim=0).values

        # Mask out padded positions
        end_idxs = torch.zeros(B, dtype=torch.long, device=device)
        best_dists = torch.zeros(B, device=device)
        for b in range(B):
            valid_len = mask[b].sum().item()
            end_idx = torch.argmin(D_table[b, Tq, 1:valid_len + 1])
            end_idxs[b] = end_idx
            best_dists[b] = D_table[b, Tq, end_idx + 1]

        best_idx = torch.argmin(best_dists).item()
        best_dist = best_dists[best_idx].item()

        # Backtrack for best candidate
        i, j = Tq, end_idxs[best_idx].item() + 1
        path_lengh = 0
        eps = 1e-4
        while i > 0:
            path_lengh += 1
            cost = dist_matrix[best_idx, i - 1, j - 1]

            # 检查是否从左上方对角线来
            if i > 0 and j > 0 and abs(D_table[best_idx, i, j] - (cost + D_table[best_idx, i - 1, j - 1])) < eps:
                i -= 1
                j -= 1
            # 检查是否从上方来
            elif i > 0 and abs(D_table[best_idx, i, j] - (cost + D_table[best_idx, i - 1, j])) < eps:
                i -= 1
            # 检查是否从左方来
            elif j > 0:
                j -= 1
            else:  # 兜底逻辑，防止死循环
                i -= 1
        start_idx = j
        end_idx = end_idxs[best_idx].item() #+ 1

        norm_dist = best_dist/path_lengh

        if dist_matrix.max() > 4.0 or norm_dist >4:
            print(f"FATAL ERROR: Distance > 4. Normalization failed! norm_dist is {norm_dist}")

        return best_idx, best_dist, start_idx, end_idx , path_lengh


def cosine_distance_matrix(x, y): #range[0,2]
    """
    Computes pairwise cosine distance between two sequences [T, D]
    Returns a [T, T] distance matrix.
    Lower = more aligned
    """
    x = torch.nn.functional.normalize(x, dim=1)
    y = torch.nn.functional.normalize(y, dim=1)
    return 1 - torch.mm(x, y.t())  # cosine similarity → distance

from scipy.special import logsumexp

def mask_sinkhorn(a, b, M, Mask, reg=0.01, numItermax=1000, stopThr=1e-9):
    # set a large value (1e6) for masked entry
    Mr = -M/reg*Mask + (-1e6)*(1-Mask)
    loga = np.log(a)
    logb = np.log(b)

    u = np.zeros(len(a))
    v = np.zeros(len(b))
    err = 1

    for i in range(numItermax):
        v = logb - logsumexp(Mr + u[:, None], 0)
        u = loga -logsumexp(Mr + v[None, :], 1)
        if i % 10 == 0:
            tmp_pi = np.exp(Mr + u[:, None] + v[None, :])
            err = np.linalg.norm(tmp_pi.sum(0) - b)
            if err < stopThr:
                return tmp_pi

    pi = np.exp(Mr + u[:, None] + v[None, :])
    return pi

def temporal_mask(T1,T2, k,d=0):
    """
    Compute temporal mask.

    """
    mask = np.zeros((T1, T2))
    for i in range(T1):
        j_center = i + d
        j_start = max(0, j_center - k)
        j_end = min(T2, j_center + k + 1)
        mask[i, j_start:j_end] = 1
    return mask
    #return np.triu(np.tril(np.ones((T, T)), k=k), k=-k)


def get_retriever(
    retrieve_key,
    state_num,
    metric,
    traj_metric,  # 'ot','sdtw'
    re_history_len,
    retrieve_len):
    retiever = Retriever(
        retrieve_key=retrieve_key,
        state_num=state_num,
        metric=metric,
        traj_metric=traj_metric,  # 'ot','sdtw'
        re_history_len=re_history_len,
        retrieve_len=retrieve_len
    )

    return retiever

'''
re_history_len = 5
import pickle as pkl
datapath='/home/carol/Project/4-RegIC_IL/expert_demos/libero/libero_90/KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet.pkl'
data = pkl.load(open(datapath, "rb"))
retiever = get_retriever(
    retrieve_key='DINO',
    state_num=5,
    metric='l2',
    traj_metric='sdtw',  # 'ot','sdtw'
    re_history_len=re_history_len,
    retrieve_len=5,
)

obs = data['observations']
ep_idx = 0
state_idx = 58
current_traj = obs[ep_idx]['retrieve_feature'][state_idx - re_history_len - 10: state_idx+1] # dict{'dino','clip'}


state_subset = retiever.get_state_subset(current_traj,obs,ex_id=ep_idx)[0][:10]
retrieve_traj_idx,retrieve_state_idx = retiever.get_traj_index_from_subset(current_traj,state_subset,obs) #traj_idx, state_idx
print(retrieve_traj_idx)


for i in np.random.randint(4, 39, size=3):
    retiever.start_traj()
    retiever.save_traj = True

    for j in np.random.randint(40, 100, size=3):
        j = 10
        print(f'traj:{i}, state:{j}: ')
        demo_id = f'demo_{i}'
        current_traj,_= retiever.dataset.get_sequence_from_demo(demo_id, j, key='obs',seq_length=10, direction=1) #10,128,128,3
        #dino,_ = retiever.dataset.get_sequence_from_demo(15, 26, key='DINOv2',seq_length=1, direction=1)
        #embed = retiever.dataset.get_embed_sequence(15, 26, seq_length=1, direction=1)
        #print(f'traj length: {len(current_traj)}')

        #state_embed = retiever.state_encode(current_traj) #10,768
        #print((dino == state_embed).all().any())
        #best_index, best_cost = find_most_similar_expert_dtw(current_traj, expert_trajs)



        for k,state in enumerate(current_traj):
            dict_guide,best_index,best_index_dtw = retiever.get_guidance(state,ex_traj_id=demo_id)

            #print(f'traj length: {len(retiever.current_emb_traj)}')

            print(f'=========== best idx is {best_index}, ')
            print(f'=========== best best_index_dtw is {best_index_dtw}, ')

'''
