import numpy as np
import os

def combine_npz(npz_list, seq_len, slot_dim, data_name):
    # import pdb; pdb.set_trace()
    num_obj = len(npz_list)
    new_X = np.zeros((num_obj, len(npz_list[0]['arr_0']), 64, 64, 3))
    new_Z = np.zeros((len(npz_list[0]['arr_1']) * seq_len, num_obj, slot_dim))
    new_num_obj = np.zeros(len(npz_list[0]['arr_2']))
    for i in range(len(npz_list)):
        X, Z, num_obj = np.array(npz_list[i]['arr_0']), npz_list[i]['arr_1'], np.array(npz_list[i]['arr_2'])
        new_X[i] = X
        new_Z[:, i, :] = Z.squeeze(2).reshape(len(npz_list[0]['arr_1']) * seq_len, slot_dim)
        new_num_obj += 1
    num_frames = new_Z.shape[0]
    # import pdb; pdb.set_trace()
    new_X = new_X.reshape((num_frames, int(new_num_obj[0]), 64, 64, 3))
    os.makedirs(data_name, exist_ok=True)
    np.savez_compressed(data_name, new_X, new_Z, new_num_obj)




if __name__=="__main__":
    # import pdb; pdb.set_trace()
    object1_npz = np.load('1_obj_sprites_still_square.npz')
    object2_npz = np.load('1_obj_sprites_moving_square.npz')

    npz_list = [object1_npz, object2_npz]
    combine_npz(npz_list, 8, 6, '2_obj_sprites_new_observations')