import torch
from timm import create_model
from torchvision import transforms, datasets
import lightning as L

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class LitClassification(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = create_model('resnet34', num_classes=196)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def training_step(self, batch):
        images, targets = batch
        outputs = self.model(images)
        loss = self.loss_fn(outputs, targets)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=0.005)
    
    
class ClassificationData(L.LightningDataModule):

    def train_dataloader(self):
        train_dataset = datasets.StanfordCars(root=".", download=False, transform=DEFAULT_TRANSFORM)
        return torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=5)
    
if __name__ == "__main__":
    model = LitClassification()
    data = ClassificationData()
    trainer = L.Trainer(max_epochs=20)
    trainer.fit(model, data)
