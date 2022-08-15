import numpy as np
import pandas as pd
from astropy.convolution import Gaussian1DKernel, convolve
import csv
import hdfdict
import os

preycap_dir = "/Users/nightcrawler/Dropbox/PreyCap_Paper/PreyCapProjectData/"
gkern = Gaussian1DKernel(2)


def find_hunted_prey(fish_id, conditions):
    
    stimuli_df = pd.read_csv(preycap_dir + fish_id + "/stimuli.csv")
    bout_df = pd.read_csv(preycap_dir + fish_id + "/huntingbouts.csv")
    stim_df_by_condition = stimuli_df[stimuli_df["Hunted Or Not"].isin(conditions)]
    bouts_per_hunt = bout_df[bout_df["Hunt ID"].isin(stim_df_by_condition["Hunt ID"])]
    unique_hunts = np.unique(bouts_per_hunt["Hunt ID"])
    unique_prey = [np.unique(stim_df_by_condition[
        stim_df_by_condition["Hunt ID"] == h]["Para ID"])[0] for h in unique_hunts]
    wrth_records = [np.load(preycap_dir + fish_id + "/wrth" + str(h).zfill(2) + ".npy") for h in unique_hunts]
    prey_rec_per_hunt = [np.array([t[up] for t in wrth]) for wrth, up in zip(wrth_records, unique_prey)]
    prey_dict_per_hunt = [{'x': np.around(convolve(r[:, 0], gkern, preserve_nan=True), decimals=6), 
                           'y': np.around(convolve(r[:, 1], gkern, preserve_nan=True), decimals=6), 
                           'z': np.around(convolve(r[:, 2], gkern, preserve_nan=True), decimals=6), 
                           'az': np.around(convolve(r[:, 6], gkern, preserve_nan=True), decimals=6), 
                           'alt': np.around(convolve(r[:, 7], gkern, preserve_nan=True), decimals=6), 
                           'dist': np.around(convolve(r[:, 4], gkern, preserve_nan=True), decimals=6),
                           'Hunt ID': [hid for r in r[:,0]]}
                          for r, hid in zip(prey_rec_per_hunt, unique_hunts)]
 
    for hi, uh in enumerate(unique_hunts):
        az_coords_at_bout = np.around(bouts_per_hunt[bouts_per_hunt["Hunt ID"] == uh]["Para Az"], decimals=6)
        az_coords_60Hz = prey_dict_per_hunt[hi]['az']
        bout_inds = [np.where(az_coords_60Hz == az)[0] for az in az_coords_at_bout]
        bout_inds_dig = [0 if i not in bout_inds else 1 for i in range(len(az_coords_60Hz))]
        prey_dict_per_hunt[hi] = {**prey_dict_per_hunt[hi], **{"BoutInds": bout_inds_dig}}
    return bouts_per_hunt, prey_rec_per_hunt, prey_dict_per_hunt        
    # got everything. now have to find the bouts. 

def save_to_hdf5(mydict):
    savepath = "/Users/nightcrawler/SpikingInferenceCircuits.jl/experiments/3DTracking/old/prey_coords.h5"
    try:
        hdfdict.dump(mydict, savepath)
    except RuntimeError:
        print("overwriting old coordfile")
        os.remove(savepath)
        return save_to_hdf5(mydict)

if __name__ == '__main__':
    p_rec = 5
    bph, prey_rec, prey_dict = find_hunted_prey("090518_5", [1, 2])
    save_to_hdf5(prey_dict[p_rec])


    
    



# Write a test function here to see how similar the very last delta az call is to the delta az calculated
# by immobilizing the fish's head vector. Just see in general what the best div is. 



# note that you calculated the az alt and dist velocities independently because you established a
# firm heading vector average and removed fish movements from the calculation (i.e. how is the fish actually perceiving
# motion?). the idea of line 2829 is to say "this is where the fish's head was when the bout began, and this is how the
# para were moving relative to that vector". para int win is 5 frames, which is ~ 80 milliseconds of integration time. 


#dist = [pr[4] for pr in poi_wrth]
    #    az = [pr[6] for pr in poi_wrth]
   #     alt = [pr[7] for pr in poi_wrth]
# kernel is kernel = Gaussian1DKernel(filter_sd) where filter_sd = 1 it seems

#filt_az = convolve(az, kernel, preserve_nan=True)
#filt_alt = convolve(az, kernel, preserve_nan=True)

