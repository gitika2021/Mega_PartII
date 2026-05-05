import numpy as np
import sys

def split_train_vald(
    lc_path,
    img_path,
    train_frac=0.8,
    seed=42,
    out_prefix="",
    out_sufix=""
):
    """
    Split paired LC and image datasets into train and vald .npy files.

    Parameters
    ----------
    lc_path : str
        Path to full light-curve .npy file
    img_path : str
        Path to full image .npy file
    train_frac : float
        Fraction of data to use for training (default 0.8)
    seed : int
        Random seed for reproducibility
    out_prefix : str
        Optional output prefix or directory (e.g. 'LC10/')
    """

    rng = np.random.default_rng(seed)

    # load data
    lc_data  = np.load(lc_path)
    img_data = np.load(img_path)

    assert len(lc_data) == len(img_data), "LC and IMG sizes do not match"

    N = len(lc_data)
    indices = rng.permutation(N)

    n_train = int(train_frac * N)

    train_idx = indices[:n_train]
    vald_idx  = indices[n_train:]

    # split
    train_lc  = lc_data[train_idx]
    vald_lc   = lc_data[vald_idx]

    train_img = img_data[train_idx]
    vald_img  = img_data[vald_idx]

    # save
    np.save(f"{out_prefix}/LC10/train_{out_sufix}LC.npy", train_lc)
    np.save(f"{out_prefix}/LC10/val_{out_sufix}LC.npy",  vald_lc)

    np.save(f"{out_prefix}/OM10/train_{out_sufix}.npy", train_img)
    np.save(f"{out_prefix}/OM10/val_{out_sufix}.npy",  vald_img)

    print(f"Saved train samples: {len(train_lc)}")
    print(f"Saved vald samples:  {len(vald_lc)}")

if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            N = str(sys.argv[1])
            # lc_path = f"/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/Data/LC10/{N}LC_processed.npy"
            # img_path = f"/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/Data/OM10/{N}.npy"

            lc_path = f"/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/Data/LC10/{N}LC_hscaled_processed.npy"
            img_path = f"/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/Data/OM10/{N}.npy"
            
            split_train_vald( lc_path,img_path,train_frac=0.8,seed=42,out_prefix="/home/iit-t/Gitika/Github-Repositories/Abraham_Mega/Reanalysis_Git/Mega_PartII_Kepler/Data/",
                            out_sufix=N)
        else:
            print('give arg: python3 genlc.py [N] [n]')
    except Exception as e:
        print(f"Main Execution Error: {e}")
