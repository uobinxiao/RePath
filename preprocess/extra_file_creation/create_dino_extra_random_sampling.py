import glob
import h5py
import os
import numpy
from tqdm import tqdm

def create_entry_dict():
    base_dir = "__REPATH_PRIVATE_PROJECT_ROOT_003__/TCGA_processed_40_256_processed/40x_256px_0px_overlap/40x_hsv_filtered_patches/"
    h5_list = glob.glob(os.path.join(base_dir, "*.h5"))

    base_dir = "__REPATH_PRIVATE_PROJECT_ROOT_003__/TCGA_processed_20_256_processed/20x_256px_0px_overlap/20x_hsv_filtered_patches/"
    h5_list = h5_list + glob.glob(os.path.join(base_dir, "*.h5"))

    base_dir = "__REPATH_PRIVATE_PROJECT_ROOT_003__/TCGA_processed_10_256_processed/10x_256px_0px_overlap/10x_hsv_filtered_patches/"
    h5_list = h5_list + glob.glob(os.path.join(base_dir, "*.h5"))

    base_dir = "__REPATH_PRIVATE_PROJECT_ROOT_003__/TCGA_processed_5_256_processed/5x_256px_0px_overlap/5x_hsv_filtered_patches/"
    h5_list = h5_list + glob.glob(os.path.join(base_dir, "*.h5"))

    all_data = []
    for h5_path in tqdm(h5_list):
        slide_name = os.path.basename(h5_path)
        slide_name = slide_name.replace("_patches.h5", ".svs")
        with h5py.File(h5_path, 'r') as hf:
            coord_data = hf["coords"][:]
            level = hf.attrs["level"]
            if coord_data.shape[0] == 0:
                continue
            slide_col = numpy.full((coord_data.shape[0], 1), slide_name)
            slide_level = numpy.full((coord_data.shape[0], 1), level)
            coord_data = numpy.hstack([coord_data.astype(object), slide_col, slide_level])

            if coord_data.shape[0] > 5000:
                idx = numpy.random.choice(coord_data.shape[0], size=5000, replace=False)
                coord_data = coord_data[idx]
            #else:
            #    print(h5_path, ":" , coord_data.shape)
            all_data.append(coord_data)

    all_data = numpy.vstack(all_data)   # (sum(N_i), 3)
    numpy.save("entries-TRAIN_v2.npy", all_data, allow_pickle=True)

def load_entry_dict():
    #data = numpy.load("entries-TRAIN.npy", mmap_mode="r")
    #print(data[1])

    data = numpy.load("entries-TRAIN_v2.npy", allow_pickle=True)
    print(data.shape)

if __name__ == "__main__":
    #create_entry_dict()
    load_entry_dict()
