from src import EPOCHS, LEARNING_RATE, BATCH_SIZE, WEIGHT_DECAY
import torch

print(f'Device Available: {torch.cuda.is_available()}')

class Train:
    def __init__(self):
        self.optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
