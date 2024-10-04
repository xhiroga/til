import random
from logging import basicConfig, getLogger, root

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from constants import symbols
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR100
from tqdm import tqdm

import wandb

basicConfig(level="INFO")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root.debug(f"{device=}")


# > We apply to each image a forward-pass through the pretrained VGG ConvNet (Simonyan & Zisserman, 2014), and represent it with the activations from either the top 1000-D softmax layer (sm) or the second-to-last 4096-D fully connected layer (fc).
class ImageEncoder:
    def __init__(self, use_softmax=True):
        self.logger = getLogger(self.__class__.__name__)
        self.vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).to(device)
        if use_softmax:
            self.softmax = torch.nn.Softmax(dim=1)
            self.output_dim = 1000
        else:
            self.vgg.classifier = self.vgg.classifier[:-1]
            self.output_dim = 4096
        self.vgg.eval()

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if image.dim() == 3:
                image = image.unsqueeze(0)
            features = self.vgg(image)
            self.logger.debug(f"{features.shape=}")  # (batch_size, embedding_dim)
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
        num_members: int = 2,
    ):
        super().__init__()
        self.logger = getLogger(self.__class__.__name__)
        self.temperature = temperature
        self.num_images = num_members

        # "The informed sender also first embeds the images into a 'game-specific' space."
        self.embedding = nn.Linear(input_dim, embed_dim)

        # 埋め込みをチャンネルとして扱い、1次元畳み込みを適用する
        # "It then applies 1-D convolutions ('filters') on the image embeddings by treating them as different channels."
        # "The informed sender uses convolutions with kernel size 2x1 applied dimension-by-dimension to the two image embeddings"
        self.conv1 = nn.Conv1d(embed_dim, num_filters, kernel_size=num_members)

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
        self.logger.debug(f"{embedded.shape=}")  # (batch_size, num_images, embed_dim)

        permuted = embedded.permute(0, 2, 1)
        self.logger.debug(f"{permuted.shape=}")  # (batch_size, embed_dim, num_images)

        conv1_out = self.conv1(permuted)
        self.logger.debug(f"{conv1_out.shape=}")  # (batch_size, num_filters, 1)

        sigmoid_out = self.sigmoid(conv1_out)
        sigmoid_out = sigmoid_out.transpose(1, 2)
        self.logger.debug(f"{sigmoid_out.shape=}")  # (batch_size, 1, num_filters)

        scores = self.conv2(sigmoid_out)
        self.logger.debug(f"{scores.shape=}")  # (batch_size, vocab_size, 1)

        squeezed = scores.squeeze(2)
        self.logger.debug(f"{squeezed.shape=}")  # (batch_size, vocab_size)

        # "a single symbol s is sampled from the resulting probability distribution."
        # もしかしたら訓練中は hard=False にしたほうが良いかも？
        gumbel = F.gumbel_softmax(squeezed / self.temperature, tau=1, hard=True).int()
        self.logger.debug(f"{gumbel.shape=}")  # (batch_size, vocab_size)
        self.logger.debug(f"{gumbel=}")

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
        self.logger = getLogger(self.__class__.__name__)
        self.temperature = temperature

        # "It embeds the images and the symbol into its own 'game-specific' space."
        self.image_embedding = nn.Linear(input_dim, embed_dim)
        self.symbol_embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, images: torch.Tensor, symbol: torch.Tensor):
        # images: (batch_size, num_images, input_dim)
        # symbol: (batch_size, vocab_size)
        self.logger.debug(f"{images.shape=}, {images.dtype=}")
        self.logger.debug(f"{symbol.shape=}, {symbol.dtype=}, {symbol=}")

        # Convert one-hot to index
        symbol_index = symbol.argmax(dim=1)

        embedded_images = self.image_embedding(images)
        embedded_symbol = self.symbol_embedding(symbol_index)
        self.logger.debug(
            f"{embedded_images.shape=}"
        )  # (batch_size, num_images, embed_dim)
        self.logger.debug(f"{embedded_symbol.shape=}")  # (batch_size, embed_dim)

        # "It then computes dot products between the symbol and image embeddings."
        similarities = torch.bmm(embedded_images, embedded_symbol.unsqueeze(2)).squeeze(
            2
        )
        self.logger.debug(
            f"{similarities.shape=}, {similarities=}"
        )  # (batch_size, num_images)

        # "The two dot products are converted to a Gibbs distribution (with temperature τ)"
        prob = F.softmax(similarities / self.temperature, dim=1)
        self.logger.debug(f"{prob.shape=}, {prob=}")

        return prob


class MultiAgentsEnvironment:
    def __init__(self, config: dict, train_loader, test_loader, vocabulary: list[str]):
        self.config = config
        self.use_softmax = config["use_softmax"]
        self.embed_dim = config["embed_dim"]
        self.num_filters = config["num_filters"]
        self.num_members = config["num_members"]
        self.vocab_size = config["vocab_size"]
        self.num_epochs = config["num_epochs"]
        self.batch_size = config["batch_size"]
        self.total_train_pairs = config["total_train_pairs"]
        self.total_test_pairs = config["total_test_pairs"]

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.encoder = ImageEncoder(use_softmax=self.use_softmax)
        self.logger = getLogger(self.__class__.__name__)
        self.sender = InformedSender(
            input_dim=self.encoder.output_dim,
            embed_dim=self.embed_dim,
            num_filters=self.num_filters,
            vocab_size=len(vocabulary),
            num_members=self.num_members,
        ).to(device)
        self.receiver = Receiver(
            input_dim=self.encoder.output_dim,
            embed_dim=self.embed_dim,
            vocab_size=len(vocabulary),
        ).to(device)
        self.optimizer = optim.Adam(
            list(self.sender.parameters()) + list(self.receiver.parameters())
        )

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        # batch_size or total_test_pairs, num_images, channel, height, width = images.shape
        batch_size, num_images, channel, height, width = images.shape
        assert num_images == self.num_members, f"{num_images=}"
        assert channel == 3, f"{channel=}"
        assert height == 224, f"{height=}"
        assert width == 224, f"{width=}"

        flattened = images.view(-1, channel, height, width)
        assert flattened.shape == (batch_size * num_images, channel, height, width)

        encoded = self.encoder.encode(flattened)

        reshaped = encoded.view(batch_size, num_images, -1)
        assert reshaped.shape == (batch_size, num_images, self.encoder.output_dim)

        return reshaped.to(device)

    def shuffle_encoded(
        self, encoded: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # encoded: (batch_size or total_test_pairs, num_images, input_dim)
        batch_size, num_images, _ = encoded.shape
        assert num_images == self.num_members, f"{num_images=}"

        indices = torch.stack(
            [torch.randperm(num_images, device=device) for _ in range(batch_size)]
        )
        shuffled = torch.stack([encoded[i][indices[i]] for i in range(batch_size)])
        target_indices = torch.argmin(indices, dim=1)

        return shuffled, indices, target_indices

    def calc_loss(
        self, prob: torch.Tensor, target_indices: torch.Tensor
    ) -> torch.Tensor:
        label = torch.zeros_like(prob)
        batch_indices = torch.arange(prob.size(0))
        label[batch_indices, target_indices] = 1.0
        self.logger.debug(f"{label.shape=}, {label=}")
        loss = F.binary_cross_entropy(prob, label)
        self.logger.debug(f"{loss.shape=}. {loss=}")
        return loss

    def train(self):
        self.sender.train()
        self.receiver.train()

        for epoch in tqdm(range(self.num_epochs)):
            epoch_loss = 0
            correct_predictions = 0

            for batch in self.train_loader:
                self.optimizer.zero_grad()

                images, _ = batch
                images = images.to(device)
                images = images.view(self.batch_size, self.num_members, 3, 224, 224)

                encoded = self.encode_images(images)

                message = self.sender(encoded)

                shuffled, _, target_indices = self.shuffle_encoded(encoded)

                prob = self.receiver(shuffled, message)

                loss = self.calc_loss(prob, target_indices)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * self.batch_size
                correct_predictions += (
                    (prob.argmax(dim=1) == target_indices).sum().item()
                )

            avg_loss = epoch_loss / self.total_train_pairs
            accuracy = correct_predictions / self.total_train_pairs
            self.logger.info(f"{epoch=}, {avg_loss=}, {accuracy=}")
            wandb.log({"epoch": epoch, "avg_loss": avg_loss, "accuracy": accuracy})

    def evaluate(self):
        with torch.no_grad():
            epoch_loss = 0
            correct_predictions = 0

            for batch in self.test_loader:
                images, _ = batch
                images = images.to(device)
                images = images.view(self.batch_size, self.num_members, 3, 224, 224)

                encoded = self.encode_images(images)

                self.logger.debug(f"{encoded.shape=}")

                message = self.sender(encoded)
                self.logger.debug(f"{message.shape=}")

                shuffled, indices, target_indices = self.shuffle_encoded(encoded)

                prob = self.receiver(shuffled, message)
                self.logger.debug(f"{prob.shape=}, {prob=}")

                shuffled, indices, target_indices = self.shuffle_encoded(encoded)
                self.logger.debug(f"{shuffled.shape=}")
                self.logger.debug(f"{indices.shape=}, {indices=}")
                self.logger.debug(f"{target_indices.shape=}, {target_indices=}")

                prob = self.receiver(shuffled, message)
                self.logger.debug(f"{prob.shape=}, {prob=}")

                loss = self.calc_loss(prob, target_indices)
                self.logger.debug(f"{loss.shape=}, {loss=}")

                batch_correct_predictions = (
                    (prob.argmax(dim=1) == target_indices).sum().item()
                )
                self.logger.debug(f"{batch_correct_predictions=}")

                epoch_loss += loss.item() * self.batch_size
                correct_predictions += batch_correct_predictions

            avg_loss = epoch_loss / self.total_test_pairs
            accuracy = correct_predictions / self.total_test_pairs
            self.logger.info(f"{avg_loss=}, {accuracy=}")
            return (avg_loss, accuracy)

    def save(self):
        pass


def get_subset(dataset, num_samples):
    indices = random.sample(range(len(dataset)), num_samples)
    return Subset(dataset, indices)


config = {
    "use_softmax": False,
    "embed_dim": 50,
    "num_filters": 20,
    "num_members": 2,
    # > We explore two vocabulary sizes: 10 and 100 symbols.
    "vocab_size": 10,
    "num_epochs": 50,
    "batch_size": 50,
    "total_train_pairs": 5000,
    "total_test_pairs": 1000,
}


def main():
    wandb.init(project="language-learning", config=config)

    # VGG16の入力サイズに合わせる
    transform = transforms.Compose(
        [transforms.Resize(size=(224, 224)), transforms.ToTensor()]
    )

    train_dataset = CIFAR100(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = CIFAR100(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        get_subset(train_dataset, config["total_train_pairs"] * config["num_members"]),
        batch_size=config["batch_size"] * config["num_members"],
        shuffle=True,
    )
    test_loader = DataLoader(
        get_subset(test_dataset, config["total_test_pairs"] * config["num_members"]),
        batch_size=config["batch_size"] * config["num_members"],
        shuffle=False,
    )

    env = MultiAgentsEnvironment(
        config=config,
        train_loader=train_loader,
        test_loader=test_loader,
        vocabulary=symbols[:10] if config["vocab_size"] == 10 else symbols,
    )

    env.train()

    test_loss, test_accuracy = env.evaluate()

    wandb.log({"test_loss": test_loss, "test_accuracy": test_accuracy})
    wandb.finish()


if __name__ == "__main__":
    main()
