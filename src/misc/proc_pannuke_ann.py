import os
import numpy as np
import cv2
import scipy.io as sio


def transform(input_dir, output_dir, folds, use_opencv=True, fix_masks=True):
    for fold in folds:
        data = np.load(os.path.join(input_dir, fold, 'Arrays', 'images.npy')) # f'{input_dir}/{fold}/Arrays/images.npy'
        masks = np.load(os.path.join(input_dir, fold, 'Masks', 'masks.npy'))
        count_img = np.shape(data)[0]
        assert count_img == np.shape(masks)[0]

        for img in range(count_img):
            if (use_opencv):
                imageCV = cv2.cvtColor(np.float32(data[img]), cv2.COLOR_BGR2RGB)
                cv2.imwrite(os.path.join(output_dir, fold, 'Images', f'{fold}_{img}.png'), imageCV) # f'{output_dir}/{fold}/Images/{fold}_{img}.png', imageCV
            if (fix_masks):
                # switch 0 and 5 types
                for xpix in range(np.shape(masks[img])[0]):
                    for ypix in range(np.shape(masks[img])[1]):
                        masks[img][xpix][ypix][0], masks[img][xpix][ypix][5] = masks[img][xpix][ypix][5], masks[img][xpix][ypix][0]
                # fix 6 channel types to instances, types
                result = np.zeros((np.shape(masks[img])[0], np.shape(masks[img])[1], 2))
                for i in range(1, np.shape(masks[img])[2]): # skip filling background with 1's
                    # instance info
                    type_channel_pannuke = masks[img][:,:,i]
                    for val in np.unique(type_channel_pannuke):
                        if val != 0.0:
                            result[:,:,0][np.where(type_channel_pannuke == val)] = val
                    # type info
                    result[:,:,1][np.where(type_channel_pannuke > 0)] = i
                    
            # remap labels
            for num,val in enumerate(np.unique(result[:,:,0])):
                result[:,:,0][result[:,:,0] == val] = float(num)
            
            # create inst type
            inst_type = []
            for i in np.unique(np.array(result[:,:,0]))[1:]:
                fpos = list(zip(*np.where(result[:,:,0] == int(i))))[0]
                inst_type.append(result[:,:,1][fpos[0]][fpos[1]])
            
            # compute instance centroids
            
            inst_centroid_list = []
            inst_map = result[:,:,0]
            inst_id_list = list(np.unique(inst_map))
            for inst_id in inst_id_list[1:]: # avoid 0 i.e background
                mask = np.array(inst_map == inst_id, np.uint8)
                inst_moment = cv2.moments(mask)
                inst_centroid = [(inst_moment["m10"] / inst_moment["m00"]), (inst_moment["m01"] / inst_moment["m00"])]
                inst_centroid_list.append(inst_centroid) 
            
            #'inst_map', 'type_map', 'inst_type', 'inst_centroid'
            result_dict = {'inst_map': result[:,:,0], 'type_map': result[:,:,1], 'inst_type': np.array([inst_type]).T, 'inst_centroid': np.array(inst_centroid_list)} 
            np.save(os.path.join(output_dir, fold, 'Labels', f'{fold}_{img}.npy'), result_dict)
            print (f'{fold}_{img}.npy saved')
            

if __name__ == "__main__":
    folds_to_use = ['test', 'valid', 'train']
    transform('/data/input/data_pannuke/', '/data/input/data_hv_pannuke/', folds_to_use)