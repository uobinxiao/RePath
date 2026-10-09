import numpy as np
import json

def convert_with_id_mapping_plain(src_npy, dst_npy, mapping_json):
    """
    Convert [x, y, filename, level] rows into a regular N x 4 array:
      - Replace each filename with an integer ID.
      - Save the ID-to-filename mapping as JSON.
    """
    obj = np.load(src_npy, allow_pickle=True)  # (N,4)
    if obj.ndim != 2 or obj.shape[1] != 4:
        raise ValueError(f"Expect (N,4) array; got shape={obj.shape}")
    N = obj.shape[0]

    xs = obj[:, 0].astype(np.int32, copy=False)
    ys = obj[:, 1].astype(np.int32, copy=False)
    lv = obj[:, 3].astype(np.int16, copy=False)
    fnames = obj[:, 2].astype(str)

    unique_fnames, inv = np.unique(fnames, return_inverse=True)
    id_to_fname = {int(i): str(name) for i, name in enumerate(unique_fnames)}

    # Assemble an N x 4 matrix directly.
    out = np.empty((N, 4), dtype=np.int32)
    out[:, 0] = xs
    out[:, 1] = ys
    out[:, 2] = inv.astype(np.int32)  # ID corresponding to each filename
    out[:, 3] = lv.astype(np.int32)

    np.save(dst_npy, out)

    with open(mapping_json, "w", encoding="utf-8") as f:
        json.dump(id_to_fname, f, ensure_ascii=False, indent=2)

    print(f"Conversion complete: {src_npy} -> {dst_npy}, shape={out.shape}")
    print(f"Filename mapping saved to: {mapping_json}, total: {len(unique_fnames)} files")

def load_test(npy_path, json_path, mmap_mode = "r"):
    arr = np.load(npy_path, mmap_mode=mmap_mode)
    with open(json_path, "r", encoding="utf-8") as f:
        id2fname = json.load(f)

    x, y, fid, level = arr[0]
    fname = id2fname[str(int(fid))]
    print(x, y, fname, level)

if __name__ == "__main__":
    src_npy = "entries-TRAIN_v2_backup.npy"
    dst_npy = "entries-TRAIN_v2.npy"
    mapping_json = "fname_meta_v2.json"
    convert_with_id_mapping_plain(src_npy, dst_npy, mapping_json)
    #load_test(dst_npy, mapping_json)
