import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from constants import symbols
from logging import basicConfig, root


basicConfig(level="DEBUG")


# > We apply to each image a forward-pass through the pretrained VGG ConvNet (Simonyan & Zisserman, 2014), and represent it with the activations from either the top 1000-D softmax layer (sm) or the second-to-last 4096-D fully connected layer (fc).
class ImageEncoder:
    def __init__(self, use_softmax=True):
        self.vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        if use_softmax:
            self.softmax = torch.nn.Softmax(dim=1)
            self.output_dim = 1000
        else:
            self.vgg.classifier = self.vgg.classifier[:-1]
            self.output_dim = 4096
        self.vgg.eval()

    def encode(self, image: torch.Tensor):
        with torch.no_grad():
            if image.dim() == 3:
                image = image.unsqueeze(0)
            return self.vgg(image)


# > The agnostic sender is a generic neural network that maps the original image vectors onto a “gamespecific” embedding space (in the sense that the embedding is learned while playing the game) followed by a sigmoid nonlinearity. Fully-connected weights are applied to the embedding concatenation to produce scores over vocabulary symbols.
class AgnosticSender:
    pass


class InformedSender(nn.Module):
    # "We set the following hyperparameters without tuning: embedding dimensionality: 50"
    # "number of filters applied to embeddings by informed sender: 20"
    # "temperature of Gibbs distributions: 10"
    # "We explore two vocabulary sizes: 10 and 100 symbols."
    def __init__(
        self,
        input_dim,
        embed_dim: int = 50,
        num_filters: int = 20,
        temperature: float = 10,
        vocab_size: int = 10,
        num_images: int = 2,
    ):
        super().__init__()
        self.temperature = temperature

        # "The informed sender also first embeds the images into a 'game-specific' space."
        self.embedding = nn.Linear(input_dim, embed_dim)

        # 埋め込みをチャンネルとして扱い、1次元畳み込みを適用する
        # "It then applies 1-D convolutions ('filters') on the image embeddings by treating them as different channels."
        # "The informed sender uses convolutions with kernel size 2x1 applied dimension-by-dimension to the two image embeddings"
        self.conv1 = nn.Conv1d(embed_dim, num_filters, kernel_size=num_images)

        # "This is followed by the sigmoid nonlinearity."
        self.sigmoid = nn.Sigmoid()

        # "The resulting feature maps are combined through another filter (kernel size f x1, where f is the number of filters on the image embeddings), to produce scores for the vocabulary symbols."
        self.conv2 = nn.Conv1d(1, vocab_size, kernel_size=num_filters)

    def forward(self, images: list[torch.Tensor]):
        # images: (batch_size, num_images, input_dim)
        assert len(images) == self.num_images

        # Note: images[0] is target image, images[1:] is distractor images
        embedded = [self.embedding(img) for img in images]
        stacked = torch.stack(embedded)
        root.debug(f"{stacked.shape=}")  # (batch_size, num_images, embed_dim)

        permuted = stacked.permute(0, 2, 1)
        root.debug(f"{permuted.shape=}")  # (batch_size, embed_dim, num_images)

        conv1_out = self.conv1(permuted)
        root.debug(f"{conv1_out.shape=}")  # (batch_size, num_filters, 1)

        sigmoid_out = self.sigmoid(conv1_out)
        root.debug(f"{sigmoid_out.shape=}")  # (batch_size, num_filters, 1)

        scores = self.conv2(sigmoid_out)
        root.debug(f"{scores.shape=}")  # (batch_size, vocab_size, 1)

        squeezed = scores.squeeze(2)
        root.debug(f"{squeezed.shape=}")  # (batch_size, vocab_size)

        prob = F.softmax(scores / self.temperature, dim=-1)

        # "a single symbol s is sampled from the resulting probability distribution."
        symbols = torch.multinomial(prob, num_samples=1)
        root.debug(f"{symbols.shape=}")  # (batch_size, 1)

        return symbols


class Receiver(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size):
        super().__init__()
        self.fc1 = nn.Linear(input_dim * 2 + vocab_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, message, img1, img2):
        x = torch.cat([message, img1, img2], dim=1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


# Envに改名するかも？
class Agents:
    def __init__(self, vocabulary: list[str]):
        #
        self.encoder = ImageEncoder(use_softmax=False)
        self.sender = InformedSender(
            input_dim=self.encoder.output_dim,
            embed_dim=50,
            num_filters=20,
            vocab_size=len(vocabulary),
        )
        self.receiver = Receiver()

    # VGG16を用いるが、VAEに変更する可能性もある
    def encode_images(self, images: list[torch.Tensor]):
        return [self.encoder.encode(img) for img in images]


def load_image(path: str):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    img = Image.open(path).convert("RGB")
    img_tensor = transform(img)
    return img_tensor


def calc_loss():
    pass


# > We explore two vocabulary sizes: 10 and 100 symbols.
config = {
    "vocab_size": 10,
}

symbols_10 = symbols[:10]
symbols_100 = symbols


def main():
    # 2^3 = 8通り
    # 1. informed sender or agnostic sender
    # 2. visual representationのsoftmax or fc
    # 3. voc size
    agents = Agents(
        vocabulary=symbols_10 if config["vocab_size"] == 10 else symbols_100
    )

    data = "./language-learning/data"
    cats = [f"{data}/images/cat/{i}.jpg" for i in range(12)]
    dogs = [f"{data}/images/dog/{i}.jpg" for i in range(1, 10)]
    cat = random.choice(cats)
    dog = random.choice(dogs)
    root.debug(f"cat: {cat}, dog: {dog}")

    cat_image = load_image(cat)
    dog_image = load_image(dog)
    images = [cat_image, dog_image]

    # このコードだと InformedSender 限定になる
    encoded_images = agents.encode_images(images)
    target, distractor = encoded_images[0], encoded_images[1]

    message = agents.sender(target, distractor)

    # Receiver selects a image with hint illustration (In 1st version, use ascii art)
    # > the receiver, instead, sees the two images in random order
    random.shuffle(encoded_images)
    shuffled1, shuffled2 = encoded_images[0], encoded_images[1]

    # If selected image is correct, increment the reward

    # Updating agent's weights


if __name__ == "__main__":
    main()
