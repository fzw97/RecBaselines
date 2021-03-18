import json
import argparse
import json
from collections import defaultdict
from time import mktime, strptime

import networkx as nx
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm

from utils import *


def load_and_save_yelp():
    # keys = ['user_id', 'name', 'review_count', 'yelping_since', 'useful',
    # 'funny', 'cool', 'elite', 'friends', 'fans', 'average_stars',
    # 'compliment_hot', 'compliment_more', 'compliment_profile', 'compliment_cute',
    # 'compliment_list', 'compliment_note', 'compliment_plain', 'compliment_cool',
    # 'compliment_funny', 'compliment_writer', 'compliment_photos']

    # Yelp download url: https://www.kaggle.com/yelp-dataset/yelp-dataset

    user_path = r'datasets/Yelp/yelp_academic_dataset_user.json'
    u2u_list = list()
    with open(user_path, 'rb') as f:
        for line in tqdm(f.readlines()):
            line = json.loads(line)
            uid = line['user_id']
            friends = line['friends'].split(', ')
            for fid in friends:
                u2u_list.append([uid, fid])

    rating_path = r'datasets/Yelp/yelp_academic_dataset_review.json'
    u2i_list = list()
    with open(rating_path, 'rb') as f:
        for line in tqdm(f.readlines()):
            line = json.loads(line)
            uid = line['user_id']
            iid = line['business_id']
            rate = line['stars']
            ts = int(mktime(strptime(line['date'], "%Y-%m-%d %H:%M:%S")))
            u2i_list.append([uid, iid, ts, rate])

    print('u2u =', len(u2u_list))
    print('u2i =', len(u2i_list))

    uid_map = defaultdict(int)  # str id --> int id
    iid_map = defaultdict(int)
    uid_map[0] = 0
    iid_map[0] = 0
    user_num = 1
    item_num = 1
    for i, (uid, iid, ts, rate) in tqdm(enumerate(u2i_list)):
        if uid_map[uid] == 0:
            uid_map[uid] = user_num
            user_num += 1

        if iid_map[iid] == 0:
            iid_map[iid] = item_num
            item_num += 1

        u2i_list[i] = [uid_map[uid], iid_map[iid], ts, rate]

    u2i = np.array(u2i_list, dtype=np.int)
    u2i = u2i[np.argsort(u2i[:, 0])]  # sort by user id

    new_u2u_list = list()
    for u1, u2 in u2u_list:
        new_u1, new_u2 = uid_map[u1], uid_map[u2]
        if new_u1 and new_u2:
            new_u2u_list.append([new_u1, new_u2])

    u2u = np.array(new_u2u_list, dtype=np.int)
    u2u = u2u[np.argsort(u2u[:, 0])]  # sort by u1 id

    print('min uid =', np.min(u2i[:, 0]))
    print('max uid =', np.max(u2i[:, 0]))
    print('num uid =', len(np.unique(u2i[:, 0])))

    print('min iid =', np.min(u2i[:, 1]))
    print('max iid =', np.max(u2i[:, 1]))
    print('num iid =', len(np.unique(u2i[:, 1])))

    print('min ts =', np.min(u2i[:, 2]))
    print('max ts =', np.max(u2i[:, 2]))

    print('min rate =', np.min(u2i[:, 3]))
    print('max rate =', np.max(u2i[:, 3]))
    print('num rate =', len(np.unique(u2i[:, 3])))

    print('min u1 id =', np.min(u2u[:, 0]))
    print('max u1 id =', np.max(u2u[:, 0]))
    print('num u1 id =', len(np.unique(u2u[:, 0])))

    print('min u2 id =', np.min(u2u[:, 1]))
    print('max u2 id =', np.max(u2u[:, 1]))
    print('num u2 id =', len(np.unique(u2u[:, 1])))

    print(u2i[:50])
    print(u2u[:50])

    np.savez(file='datasets/Yelp/u2ui.npz',
             u2i=u2i,
             u2u=u2u)
    np.savez(file='datasets/Yelp/uid_map.npz',
             uid_map=uid_map)

    print('saved at', 'datasets/Yelp/u2ui.npz')


def filter_and_reid():
    '''
    Raw u2i (8021100, 3)
    min user = 1
    max user = 1968703
    num user = 1968703
    min item = 1
    max item = 209393
    num item = 209393

    Raw u2u edges: 19042100

    '''

    u2ui = np.load(f'datasets/Yelp/u2ui.npz')
    u2u, u2i = u2ui['u2u'], u2ui['u2i']
    df = pd.DataFrame(data=u2i, columns=['user', 'item', 'ts', 'rate'])
    df.drop_duplicates(subset=['user', 'item', 'ts', 'rate'], keep='first', inplace=True)

    print('Raw u2i', df.shape)
    print('min user =', df['user'].min())
    print('max user =', df['user'].max())
    print('num user =', len(np.unique(df.values[:, 0])))
    print('min item =', df['item'].min())
    print('max item =', df['item'].max())
    print('num item =', len(np.unique(df.values[:, 1])))

    df = preprocess_uir(df, prepro='3filter', pos_threshold=3, level='u')
    df.drop(['rate'], axis=1, inplace=True)

    print('Processed u2i', df.shape)
    print('min user =', df['user'].min())
    print('max user =', df['user'].max())
    print('num user =', len(np.unique(df.values[:, 0])))
    print('min item =', df['item'].min())
    print('max item =', df['item'].max())
    print('num item =', len(np.unique(df.values[:, 1])))

    df = df.sort_values(['user', 'ts'], kind='mergesort').reset_index(drop=True)
    u2i = df.values

    user_idmap = defaultdict(int)  # src id -> new id
    user_idmap[0] = 0
    user_num = 1

    for i, (user, item, ts) in tqdm(enumerate(u2i)):
        if user_idmap[user] == 0:
            user_idmap[user] = user_num
            user_num += 1

        u2i[i, 0] = user_idmap[user]

    print('Raw u2u edges:', len(u2u))
    new_uu_elist = []
    for u1, u2 in tqdm(u2u):
        new_u1 = user_idmap[u1]
        new_u2 = user_idmap[u2]
        if new_u1 and new_u2:
            new_uu_elist.append([new_u1, new_u2])

    print('Processed u2u edges:', len(new_uu_elist))
    u2u = np.array(new_uu_elist).astype(np.int32)
    u2i = u2i.astype(np.int32)

    save_path = 'datasets/Yelp/reid_u2ui.npz'
    np.savez(file=save_path, u2u=u2u, u2i=u2i)

    print('saved at', save_path)


def delete_isolated_user():
    u2ub = np.load('datasets/Yelp/reid_u2ui.npz')
    uu_elist = u2ub['u2u']
    u2i = u2ub['u2i']

    print('Building u2u graph...')
    user_num = np.max(u2i[:, 0]) + 1
    g = nx.Graph()
    g.add_nodes_from(list(range(user_num)))
    g.add_edges_from(uu_elist)
    g.remove_node(0)

    isolated_user_set = set(nx.isolates(g))
    print('Isolated user =', len(isolated_user_set))

    new_u2i = []
    for user, item, ts in tqdm(u2i):
        if user not in isolated_user_set:
            new_u2i.append([user, item, ts])

    new_u2i = np.array(new_u2i, dtype=np.int32)

    print('No isolated user u2i =', new_u2i.shape)

    user_idmap = defaultdict(int)  # src id -> new id
    user_idmap[0] = 0
    user_num = 1
    for i, (user, item, ts) in tqdm(enumerate(new_u2i)):
        if user_idmap[user] == 0:
            user_idmap[user] = user_num
            user_num += 1

        new_u2i[i, 0] = user_idmap[user]

    new_uu_elist = []
    for u1, u2 in tqdm(uu_elist):
        new_u1 = user_idmap[u1]
        new_u2 = user_idmap[u2]
        if new_u1 and new_u2:
            new_uu_elist.append([new_u1, new_u2])

    new_uu_elist = np.array(new_uu_elist, dtype=np.int32)

    df = pd.DataFrame(data=new_u2i, columns=['user', 'item', 'ts'])
    df['item'] = pd.Categorical(df['item']).codes + 1

    print(df.head(20))

    # cc_sizes = [len(c) for c in sorted(nx.connected_components(g), key=len, reverse=True)]
    # print('u2u connected components sizes (top20):', cc_sizes[:20])
    # print('Isolated user =', np.sum(np.array(cc_sizes) == 1))

    user_num = df['user'].max() + 1
    item_num = df['item'].max() + 1

    print('min user =', df['user'].min())
    print('max user =', df['user'].max())
    num_user = len(np.unique(df.values[:, 0]))
    print('num user =', num_user)

    print('min item =', df['item'].min())
    print('max item =', df['item'].max())
    num_item = len(np.unique(df.values[:, 1]))
    print('num item =', num_item)

    print(f'Loaded Yelp dataset with {user_num} users, {item_num} items, '
          f'{len(df.values)} u2i, {len(new_uu_elist)} u2u. ')

    new_u2i = df.values.astype(np.int32)
    save_path = 'datasets/Yelp/noiso_reid_u2ui.npz'
    np.savez(file=save_path, u2u=new_uu_elist, u2i=new_u2i)
    return num_user, num_item


def data_partition(df):
    print('Splitting train/val/test set...')
    user_train = defaultdict(list)
    user_valid = defaultdict(list)
    user_test = defaultdict(list)

    eval_users = []
    valid_items = []
    test_items = []

    user_items_dict = defaultdict(list)

    def apply_fn1(grp):
        key_id = grp['user'].values[0]
        user_items_dict[key_id] = grp[['item', 'ts']].values

    df.groupby('user').apply(apply_fn1)
    print('Groupby user finished.')

    for user in tqdm(user_items_dict.keys()):
        nfeedback = len(user_items_dict[user])
        if nfeedback < 5:
            user_train[user] = user_items_dict[user]
        else:
            # Append user history items
            eval_users.append(user)
            user_train[user] = user_items_dict[user][:-2]

            # Second last item for validation
            valid_item = user_items_dict[user][-2][0]
            user_valid[user].append(valid_item)
            valid_items.append(valid_item)

            # Last item for test
            test_item = user_items_dict[user][-1][0]
            user_test[user].append(test_item)
            test_items.append(test_item)

    return user_train, user_valid, user_test, eval_users, valid_items, test_items


def gen_and_save_u2u_dict_and_split(num_user, num_item):
    u2ui = np.load('datasets/Yelp/noiso_reid_u2ui.npz')

    print('Building u2u graph...')
    g = nx.Graph()
    g.add_nodes_from(list(range(num_user)))
    g.add_edges_from(u2ui['u2u'])
    g.remove_node(0)

    print('To undirected graph...')
    g.to_undirected()
    # g.add_edges_from([[u, u] for u in g.nodes])
    u2u_dict = nx.to_dict_of_lists(g)

    df = pd.DataFrame(data=u2ui['u2i'], columns=['user', 'item', 'ts'])
    print('Raw u2i =', df.shape)
    df.drop_duplicates(subset=['user', 'item', 'ts'], keep='first', inplace=True)
    df = df.sort_values(['user', 'ts'], kind='mergesort').reset_index(drop=True)
    print('Processed u2i =', df.shape)

    user_train, user_valid, user_test, eval_users, valid_items, test_items = data_partition(df)

    save_path = 'datasets/Yelp/u2u_split_dicts.pkl'
    save_pkl(save_path, [
        u2u_dict, user_train, user_valid, user_test,
        eval_users, valid_items, test_items])

    print('saved at', save_path)


def get_nbr(u2u, user, nbr_maxlen):
    nbr = np.zeros([nbr_maxlen, ], dtype=np.int32)
    nbr_len = len(u2u[user])
    if nbr_len == 0:
        pass
    elif nbr_len > nbr_maxlen:
        np.random.shuffle(u2u[user])
        nbr[:] = u2u[user][:nbr_maxlen]
    else:
        nbr[:nbr_len] = u2u[user]

    return nbr


def get_nbr_iids(user_train, user, nbrs, time_splits):
    nbr_maxlen = len(nbrs)
    seq_maxlen = len(time_splits)
    nbrs_iids = np.zeros((nbr_maxlen, seq_maxlen), dtype=np.int32)

    start_idx = np.nonzero(time_splits)[0]
    if len(start_idx) == 0:
        return nbrs_iids
    else:
        start_idx = start_idx[0]

    user_first_ts = time_splits[start_idx]
    user_last_ts = time_splits[-1]

    for i, nbr in enumerate(nbrs):
        if nbr == 0 or nbr == user:
            continue

        nbr_hist = user_train[nbr]

        if len(nbr_hist) == 0:
            continue

        nbr_first_ts = nbr_hist[0][1]
        nbr_last_ts = nbr_hist[-1][1]

        if nbr_first_ts > user_last_ts or nbr_last_ts <= user_first_ts:
            continue

        sample_list = list()
        for j in range(start_idx + 1, seq_maxlen):
            start_time = time_splits[j - 1]
            end_time = time_splits[j]

            if start_time != end_time:
                sample_list = list(filter(None, map(
                    lambda x: x[0] if x[1] > start_time and x[1] <= end_time else None, nbr_hist
                )))

            if len(sample_list):
                # print('st={} et={} sl={}'.format(start_time, end_time, sample_list))
                nbrs_iids[i, j] = np.random.choice(sample_list)

    return nbrs_iids


def gen_and_save_all_user_batches(user_num, item_num):
    # eval batch for each user
    u2u_dict, user_train, user_valid, user_test, eval_users, valid_items, test_items = \
        load_pkl('datasets/Yelp/u2u_split_dicts.pkl')

    def sample_one_user(user):
        seq = np.zeros(seq_maxlen, dtype=np.int32)
        pos = np.zeros(seq_maxlen, dtype=np.int32)
        ts = np.zeros(seq_maxlen, dtype=np.int32)
        nxt = user_train[user][-1, 0]
        idx = seq_maxlen - 1

        for (item, time_stamp) in reversed(user_train[user][:-1]):
            seq[idx] = item
            ts[idx] = time_stamp
            pos[idx] = nxt
            nxt = item
            idx -= 1
            if idx == -1: break

        nbr = get_nbr(u2u_dict, user, nbr_maxlen)
        nbr_iid = get_nbr_iids(user_train, user, nbr, ts)
        nbr_iid = sp.csr_matrix(nbr_iid, dtype=np.int32)

        return user, seq, pos, nbr, nbr_iid

    uid_list = []
    seq_list = []
    pos_list = []
    nbr_list = []
    nbr_iid_list = []

    for user in tqdm(range(1, user_num)):
        user, seq, pos, nbr, nbr_iid = sample_one_user(user)
        uid_list.append(user)
        seq_list.append(seq)
        pos_list.append(pos)
        nbr_list.append(nbr)
        nbr_iid_list.append(nbr_iid)

    # save as npz
    np.savez(
        'datasets/Yelp/processed_data.npz',
        user_train=user_train,
        user_valid=user_valid,
        user_test=user_test,
        eval_users=np.array(eval_users, dtype=np.int32),
        valid_items=np.array(valid_items, dtype=np.int32),
        test_items=np.array(test_items, dtype=np.int32),
        train_uid=np.array(uid_list, dtype=np.int32),
        train_seq=np.array(seq_list, dtype=np.int32),
        train_pos=np.array(pos_list, dtype=np.int32),
        train_nbr=np.array(nbr_list, dtype=np.int32),
        train_nbr_iid=nbr_iid_list
    )

    print('saved at datasets/Yelp/processed_data.npz')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', default='Yelp')
    parser.add_argument('--edim', type=int, default=64)
    parser.add_argument('--seq_maxlen', type=int, default=50)
    parser.add_argument('--nbr_maxlen', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    seed = args.seed
    np.random.seed(seed)
    seq_maxlen = args.seq_maxlen
    nbr_maxlen = args.nbr_maxlen

    load_and_save_yelp()  # 从logs生成npz

    # 1. 筛选数据，重新标id
    # filter_and_reid()  # 5filter删除item和user -> 重新赋值uid -> 从u2u删除没有item的user
    # num_user, num_item = delete_isolated_user()  # 删除孤立用户 -> 重新赋值uid -> 更新u2u和u2b的uid

    # num_user = num_user + 1
    # num_item = num_item + 1

    # 2. Generate Training Samples
    # gen_and_save_u2u_dict_and_split(num_user, num_item)
    # gen_and_save_all_user_batches(num_user, num_item)
