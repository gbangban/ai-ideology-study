import wandb
wandb.init(project="test", config={"test": True})
wandb.log({"accuracy": 0.95})
wandb.finish()
print("W&B connection OK")
