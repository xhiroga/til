import random
from itertools import chain
from logging import basicConfig, root

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from constants import symbols

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
        one_hot: bool = False,
    ):
        super().__init__()
        self.temperature = temperature
        self.num_images = num_images
        self.one_hot = one_hot

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

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: (batch_size, num_images, input_dim)
        assert images.shape[1] == self.num_images, f"{images.shape[1]=}"

        # Note: images[:, 0] is target image, images[:, 1:] is distractor images
        embedded = self.embedding(images.view(-1, images.shape[-1])).view(
            images.shape[0], images.shape[1], -1
        )
        root.debug(f"{embedded.shape=}")  # (batch_size, num_images, embed_dim)

        permuted = embedded.permute(0, 2, 1)
        root.debug(f"{permuted.shape=}")  # (batch_size, embed_dim, num_images)

        conv1_out = self.conv1(permuted)
        root.debug(f"{conv1_out.shape=}")  # (batch_size, num_filters, 1)

        sigmoid_out = self.sigmoid(conv1_out)
        sigmoid_out = sigmoid_out.transpose(1, 2)
        root.debug(f"{sigmoid_out.shape=}")  # (batch_size, 1, num_filters)

        scores = self.conv2(sigmoid_out)
        root.debug(f"{scores.shape=}")  # (batch_size, vocab_size, 1)

        squeezed = scores.squeeze(2)
        root.debug(f"{squeezed.shape=}")  # (batch_size, vocab_size)

        # "a single symbol s is sampled from the resulting probability distribution."
        gumbel = F.gumbel_softmax(squeezed / self.temperature, tau=1, hard=self.one_hot)
        root.debug(f"{gumbel.shape=}")  # (batch_size, vocab_size)
        root.debug(f"{gumbel=}")

        return gumbel


class Receiver(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dim: int = 50,
        temperature: float = 10,
        vocab_size: int = 10,
    ):
        super().__init__()
        self.temperature = temperature

        # "It embeds the images and the symbol into its own 'game-specific' space."
        self.image_embedding = nn.Linear(input_dim, embed_dim)
        self.symbol_embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, images: torch.Tensor, symbol: torch.Tensor):
        # images: (batch_size, num_images, input_dim)
        # symbol: (batch_size, 1)

        embedded_images = self.image_embedding(images)
        embedded_symbol = self.symbol_embedding(symbol)

        # "It then computes dot products between the symbol and image embeddings."
        similarities = [
            torch.sum(embedded_symbol * img, dim=1, keepdim=True)
            for img in embedded_images
        ]

        # Stack similarities
        stacked_similarities = torch.cat(
            similarities, dim=1
        )  # (batch_size, num_images)

        # "The two dot products are converted to a Gibbs distribution (with temperature τ)"
        probs = F.softmax(stacked_similarities / self.temperature, dim=1)

        # "the receiver 'points' to an image by sampling from the resulting distribution."
        chosen_image = torch.multinomial(probs, num_samples=1)

        return chosen_image


# Envに改名するかも？
class Agents:
    def __init__(self, vocabulary: list[str], train=True):
        self.encoder = ImageEncoder(use_softmax=False)
        self.sender = InformedSender(
            input_dim=self.encoder.output_dim,
            embed_dim=50,
            num_filters=20,
            vocab_size=len(vocabulary),
            num_images=2,
            one_hot=not train,
        )
        self.receiver = Receiver(
            input_dim=self.encoder.output_dim,
            embed_dim=50,
            vocab_size=len(vocabulary),
        )

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

    # tensorに変換
    batch_images = torch.stack([target, distractor]).unsqueeze(0)  # (1, 2, input_dim)

    message = agents.sender(batch_images)

    # Receiver selects a image with hint illustration (In 1st version, use ascii art)
    # > the receiver, instead, sees the two images in random order
    random.shuffle(encoded_images)
    shuffled1, shuffled2 = encoded_images[0], encoded_images[1]

    # If selected image is correct, increment the reward

    # Updating agent's weights


if __name__ == "__main__":
    main()
