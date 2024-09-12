import random
from logging import basicConfig, root

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

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
            features = self.vgg(image)
            root.debug(f"{features.shape=}")  # (batch_size, embedding_dim)
            return features


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
        self.num_images = num_images

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
        # もしかしたら訓練中は hard=False にしたほうが良いかも？
        gumbel = F.gumbel_softmax(squeezed / self.temperature, tau=1, hard=True).int()
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
        # symbol: (batch_size, vocab_size)
        root.debug(f"{images.shape=}, {images.dtype=}")
        root.debug(f"{symbol.shape=}, {symbol.dtype=}, {symbol=}")

        # Convert one-hot to index
        symbol_index = symbol.argmax(dim=1)

        embedded_images = self.image_embedding(images)
        embedded_symbol = self.symbol_embedding(symbol_index)
        root.debug(f"{embedded_images.shape=}")  # (batch_size, num_images, embed_dim)
        root.debug(f"{embedded_symbol.shape=}")  # (batch_size, embed_dim)

        # "It then computes dot products between the symbol and image embeddings."
        similarities = torch.bmm(embedded_images, embedded_symbol.unsqueeze(2)).squeeze(
            2
        )
        root.debug(
            f"{similarities.shape=}, {similarities=}"
        )  # (batch_size, num_images)

        # "The two dot products are converted to a Gibbs distribution (with temperature τ)"
        prob = F.softmax(similarities / self.temperature, dim=1)
        root.debug(f"{prob.shape=}, {prob=}")

        return prob


# Envに改名するかも？
class Agents:
    def __init__(self, vocabulary: list[str]):
        self.encoder = ImageEncoder(use_softmax=False)
        self.sender = InformedSender(
            input_dim=self.encoder.output_dim,
            embed_dim=50,
            num_filters=20,
            vocab_size=len(vocabulary),
            num_images=2,
        )
        self.receiver = Receiver(
            input_dim=self.encoder.output_dim,
            embed_dim=50,
            vocab_size=len(vocabulary),
        )

    # TODO: 引数と返り値の型を torch.Tensor にしたほうが良いかも
    def encode_images(self, images: torch.Tensor) -> list[torch.Tensor]:
        # images: (num_images, input_dim)
        return self.encoder.encode(images)


def load_image(path: str) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    img = Image.open(path).convert("RGB")
    img_tensor = transform(img)
    return img_tensor


def calc_loss(prob: torch.Tensor, target_index: tuple):
    label = torch.zeros_like(prob)
    label[:, target_index] = 1.0
    root.debug(f"{label.shape=}, {label=}")
    loss = F.binary_cross_entropy(prob, label)
    root.debug(f"{loss.shape=}. {loss=}")
    return loss


def train_step(agents, optimizer, images: torch.Tensor):
    # images: (num_images, input_dim)
    optimizer.zero_grad()

    # Encode images
    encoded_images = agents.encode_images(images)
    root.debug(f"{encoded_images.shape=}")  # (num_images, input_dim)
    batch_images = encoded_images.unsqueeze(0)
    root.debug(f"{batch_images.shape=}")  # (1, num_images, input_dim)

    # Sender generates a message
    message = agents.sender(batch_images)

    # Receiver tries to identify the target image
    indices = torch.randperm(len(images))
    root.debug(f"{indices.shape=}, {indices=}")
    target_index = torch.where(indices == 0)
    root.debug(f"{target_index=}")
    shuffled_batch = batch_images[:, indices, :]
    prob = agents.receiver(shuffled_batch, message)

    # Calculate loss
    loss = calc_loss(prob, target_index)

    # Backpropagate and update weights
    loss.backward()
    optimizer.step()

    is_correct = prob.argmax().item() == target_index
    root.debug(f"{loss.item()=}, {is_correct=}")
    return loss.item(), is_correct


def evaluate(agents, cats, dogs, num_tests):
    correct_predictions = 0

    for _ in range(num_tests):
        cat = random.choice(cats)
        dog = random.choice(dogs)

        cat_image = load_image(cat)
        dog_image = load_image(dog)
        images = [cat_image, dog_image]

        target_index = random.randint(0, 1)

        encoded_images = agents.encode_images(images)
        batch_images = torch.stack(encoded_images).unsqueeze(0)

        with torch.no_grad():
            message = agents.sender(batch_images)
            shuffled_images = encoded_images.copy()
            random.shuffle(shuffled_images)
            shuffled_batch = torch.stack(shuffled_images).unsqueeze(0)
            chosen_image = agents.receiver(shuffled_batch, message)

        if chosen_image.argmax().item() == target_index:
            correct_predictions += 1

    return correct_predictions / num_tests


# > We explore two vocabulary sizes: 10 and 100 symbols.
config = {
    "vocab_size": 10,
    "num_epochs": 10,
    "batch_size": 2,
}


def main():
    agents = Agents(vocabulary=symbols[:10] if config["vocab_size"] == 10 else symbols)

    optimizer = optim.Adam(
        list(agents.sender.parameters()) + list(agents.receiver.parameters())
    )

    data = "./language-learning/data"
    cats = [f"{data}/images/cat/{i}.jpg" for i in [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11]]
    dogs = [f"{data}/images/dog/{i}.jpg" for i in [1, 2, 3, 5, 6, 7, 8, 9]]

    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]

    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0
        correct_predictions = 0

        for _ in range(batch_size):
            cat = random.choice(cats)
            dog = random.choice(dogs)

            cat_image = load_image(cat)
            dog_image = load_image(dog)
            images = torch.stack([cat_image, dog_image])
            random.shuffle(images)
            root.debug(f"{images.shape=}")

            loss, is_correct = train_step(agents, optimizer, images)

            epoch_loss += loss
            if is_correct:
                correct_predictions += 1

        avg_loss = epoch_loss / batch_size
        accuracy = correct_predictions / batch_size

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}: Avg Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}"
            )

    # Test the trained model
    test_accuracy = evaluate(agents, cats, dogs, num_tests=100)
    print(f"Test Accuracy: {test_accuracy:.2f}")


if __name__ == "__main__":
    main()
