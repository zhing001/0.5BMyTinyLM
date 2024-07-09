import os
from transformers import TrainerCallback, TrainerState, TrainerControl
import psutil

__all__ = [
    "MemoryUsagePrintCallback",
    "SavedPerEpochCallback"
]

class MemoryUsagePrintCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        print("Training started; Initial memory usage:")
        self.print_memory_usage()

    def on_step_begin(self, args, state, control, **kwargs):
        print(f"Step {state.global_step}; Memory usage before step execution:")
        self.print_memory_usage()

    def on_step_end(self, args, state, control, **kwargs):
        print(f"Step {state.global_step}; Memory usage after step execution:")
        self.print_memory_usage()

    def print_memory_usage(self):
        mem = psutil.virtual_memory()
        print(f"Total: {mem.total}, Available: {mem.available}, Used: {mem.used}, Percent: {mem.percent}%")


class SavedPerEpochCallback(TrainerCallback):
    def on_epoch_end(self, args, state:TrainerState, control:TrainerControl, **kwargs):
        model = kwargs['model']
        model.save_pretrained(os.path.join(args.output_dir, f'epoch_{int(state.epoch)}_saved'))