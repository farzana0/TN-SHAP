# src/extra_experiments/exp1_sparse_poly/config.py

SEED = 0

D = 50                 # input dimension
N_TRAIN = 2000
N_TEST = 500

ACTIVE_RATIO = 1/3

NOISE_STD = 0.1

TEACHER_TYPE = "mlp"   # or "svr"
N_EXPLAIN = 100        # how many test points to explain

BACKGROUND_SIZE = 100  # for KernelSHAP
KERNELSHAP_SAMPLES = 500
