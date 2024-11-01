import numpy as np

def combine_npz(npz_list, seq_len, slot_dim, data_name):
    import pdb; [pdb.set_trace()]
    num_obj = len(npz_list)
    new_X = np.zeros((num_obj, len(npz_list[0]['arr_0']), 64, 64, 3))
    new_Z = np.zeros((len(npz_list[0]['arr_1']), seq_len, num_obj, slot_dim))
    new_num_obj = np.zeros(len(npz_list[0]['arr_2']))
    for i in range(len(npz_list)):
        X, Z, num_obj = np.array(npz_list[i]['arr_0']), npz_list[i]['arr_1'], np.array(npz_list[i]['arr_2'])
        new_X[i] = X
        new_Z[:, :, 0, :] = Z.squeeze(2)
        new_num_obj += 1
    num_seq = new_Z.shape[0]
    new_X.reshape((num_seq, seq_len, len(npz_list), 64, 64, 3))
    np.savez_compressed("data/datasets/" + data_name, X, Z, num_obj)




if __name__=="__main__":
    # import pdb; pdb.set_trace()
    object1_npz = np.load('1_obj_sprites_still_square.npz')
    object2_npz = np.load('1_obj_sprites_moving_square.npz')

    npz_list = [object1_npz, object2_npz]
    combine_npz(npz_list, 8, 6, '2_obj_sprites_new_observations')