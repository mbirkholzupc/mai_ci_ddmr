import utils.config_file
import os
import shutil

def preprocess_main(cfg):
    print('Preprocessing')
    print(cfg)

    # Clear out old preprocessed dir and create fresh one
    preprocessedimgdir=cfg.outdatadir+'/data'
    if os.path.exists(preprocessedimgdir):
        shutil.rmtree(preprocessedimgdir)

    os.mkdir(preprocessedimgdir)

    # What flow do we want here?
    # I am thinking:
    # 1) Copy input images to out/data
    # 2) Augment data and save augmented data to out/data
    # 3) Load images and perform stratified k-fold split
    #      - Is split random? If so, do we want random_state for reproduceability?
    #      - Or, do we want the split to acutally just create lists of filenames? Might be useful
    #        if we don't want to split data each time, but might not be worth the hassle.
    # 4) Return sets of preprocessed images in some form that train/test can handle, whether that's
    #    text filenames or arrays of references to images

    # TODO: This return is just a placeholder. Update it with what we decide for #4
    return preprocessedimgdir

