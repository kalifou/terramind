import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import measure_flops
import time


class GPUHoursCallback(Callback):
    def __init__(self):
        super().__init__()
        self.gpu_hours = None
    def on_fit_start(self, trainer, pl_module):
        torch.cuda.synchronize()
        self._t0 = time.time()

    def on_fit_end(self, trainer, pl_module):
        torch.cuda.synchronize()
        t1 = time.time()
        duration = t1 - self._t0                     # seconds of real time
        num_gpus = max(1, trainer.strategy.world_size)
        gpu_hours = duration * num_gpus / 3600.0      # GPU-seconds → GPU-hours
        self.gpu_hours = gpu_hours
        print(f"\n→ GPU-Hours: {gpu_hours:.3f} h")

class PeakMemoryCallback(Callback):
    def __init__(self):
        super().__init__()
        self.peak_gb = None
    def on_fit_start(self, trainer, pl_module):
        torch.cuda.reset_peak_memory_stats()

    def on_fit_end(self, trainer, pl_module):
        peak_bytes = torch.cuda.max_memory_allocated()      # bytes
        peak_gb = peak_bytes / (1024**3)
        self.peak_gb = peak_gb
        print(f"→ Peak GPU memory: {peak_gb:.2f} GB")
        

class FlopsCallback(Callback):
    """
    Computes and logs model FLOPs at the end of training.
    Usage: add FlopsEndCallback(input_shape=(1,6,128,128)) to your Trainer callbacks.
    """
    def __init__(self, input_shape=(1, 6, 128, 128)):
        super().__init__()
        self.input_shape = input_shape
        self.gflops = None

    def on_fit_end(self, trainer, pl_module):   
        x = torch.randn(self.input_shape, device=pl_module.device)  # Adjust input shape as needed

        pl_module.model.eval()  # Set the model to evaluation mode
        model_fwd = lambda: pl_module.model(x)
        fwd_flops = measure_flops(pl_module.model, model_fwd)
        gflops = fwd_flops / 1e9
        print(f"→ FLOPs: {gflops:.3f} GFLOPs")
        self.gflops = gflops