import wandb
import os

os.environ["WANDB_DISABLE_SERVICE"] = "True"
os.environ['WANDB_START_METHOD'] = 'thread'


class Logger:
    def __init__(self, config):
        self.config = config
        self.use_wandb = config['logging']['use_wandb']
        self.verbose = config['logging']['verbose']
        if self.use_wandb:
            if wandb.run is not None:
                print("Found existing W&B run")
                wandb.finish()
            wandb.init(project=config['logging']['wandb_project'], config=config, reinit=True)
            print("Initialized W&B run")

    def log_metrics(self, metrics, episode):
        if self.verbose != 0 and episode % self.verbose == 0:
            print("-" * 50)
            for key, value in metrics.items():
                print(f"{key}: {value}")
        if self.use_wandb:
            wandb.log(metrics, step=episode)

    def finish(self):
        wandb.finish()

