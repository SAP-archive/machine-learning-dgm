import numpy as np
from easydict import EasyDict as edict
__C = edict()
opt = __C

__C.method = 'DGMw'
__C.dataset = 'mnist'
__C.log_dir = '/home/ec2-user/imagenet_mnt/std_dgm/logs/DGMw/std_runs/'
__C.dataroot= 'dat/split_mnist_'
__C.outf='/home/ec2-user/imagenet_mnt/std_dgm/outputs/DGMw'
__C.outf_models='outputs/DGMw/models'
__C.batchSize= 64
__C.imageSize= 32
__C.manualSeed= 2
__C.nz= 128 # size of the latent z vector
__C.ngf= 6
__C.ndf= 32
__C.niter= 251
__C.lr_D= 0.002
__C.lr_G= 0.002
__C.beta1= 0.5
__C.cuda= True
__C.device= 3
__C.manualSeed= 100
__C.smax_g= 1e+5
__C.lamb_G= 0.08
__C.lambda_adv= 1.
__C.lambda_wasserstein= 1.
__C.reinit_D = True
__C.store_models= False
__C.aux_G=False


__C.workers=1
__C.dataroot_val=''
#__C.nruns=10
__C.device_G = 'cuda:0'
__C.device_D = 'cuda:0'
#__C.gpus= '3,4,5,6,7'
#__C.nproc_gpu=2
#__C.sleep=0.9
#__C.tmp_folder='/home/ec2-user/imagenet_mnt/std_dgm/tmp/'
__C.calc_fid_imnet = False
__C.class_idx_imnet = []


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if not k in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)