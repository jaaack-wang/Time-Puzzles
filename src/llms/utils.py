def clean_cuda_cache():
    import gc
    import torch

    gc.collect()
    torch.cuda.empty_cache()