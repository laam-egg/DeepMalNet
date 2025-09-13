def select_device():
    import torch
    if torch.cuda.is_available():
        print(f"[INFO] select_device(): CUDA is available")
        device_index = torch.cuda.current_device()
        DEVICE_NAME = f"cuda:{device_index}"
    else:
        print(f"[WARNING] select_device(): CUDA is not available ; falling back to CPU")
        DEVICE_NAME = "cpu"
    return DEVICE_NAME
