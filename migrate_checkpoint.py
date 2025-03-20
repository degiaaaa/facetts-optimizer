import torch
import pytorch_lightning as pl

def migrate_checkpoint(checkpoint_path):
    try:
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Add 'pytorch-lightning_version' if it is missing
        if 'pytorch-lightning_version' not in checkpoint:
            checkpoint['pytorch-lightning_version'] = pl.__version__
            print(f"Added 'pytorch-lightning_version': {pl.__version__} to the checkpoint.")

        # Save the updated checkpoint
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint at '{checkpoint_path}' has been successfully updated.")

    except Exception as e:
        print(f"An error occurred while migrating the checkpoint: {e}")

if __name__ == "__main__":
    checkpoint_path = "./ckpts/facetts_lrs3.pt"  
    migrate_checkpoint(checkpoint_path)