import argparse
import typing as t
import inspect
from functools import wraps
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.mnist import MNIST
from torchvision.utils import make_grid, save_image


def init_weights(model: nn.Module):
    if isinstance(model, nn.Linear):
        torch.nn.init.xavier_uniform(model.weight)
        model.bias.data.fill_(0.01)


def get_concrete_optimizers() -> t.List[torch.optim.Optimizer]:
    """
    Returns: A list of concrete (non-abstract) optimizer objects
    """
    return [subclass for subclass in get_all_subclasses(torch.optim.Optimizer)
            if not inspect.isabstract(subclass)]


def get_optimizer(optimizer_name: str, *args, **kwargs) -> torch.optim.Optimizer:
    """
    A simple utility function for retrieving the desired optimizer.
    Not the most optimized algorithm, but the number of optimizers should never
    exceed even 100 in my opinion so a linear time algorithm is acceptable.
    Args:
        optimizer_name: The name of the optimizer that we wish to retrieve.
    Returns:
        An optimizer defined inside of PyTorch.
    """
    concrete_optimizers: t.List = get_concrete_optimizers()
    optimizer_names: t.List[str] = [cls.__name__.lower() for cls in concrete_optimizers]
    try:
        optimizer_index = optimizer_names.index(optimizer_name.lower())
    except ValueError:
        raise ValueError(f"Passed in undefined optimizer: {optimizer_name}. "
                         f"Available optimizers: {optimizer_names}")
    return concrete_optimizers[optimizer_index](*args, **kwargs)


def get_all_subclasses(cls):
    """
    Given an class object, get all the subclasses that exist
    Args:
        cls: The class that we wish to inspect
    Returns:

    """
    all_subclasses = []

    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))

    return all_subclasses


def gan_argparse(script_initialization_example: str,
                 parser: argparse.ArgumentParser = None):
    """
    Given a script initialization example, decorate an arg-parse returning function
    a
    Args:
        script_initialization_example:
        parser: An existing argument parser to add arguments to.
        If not defined, we will create a new argument parser.

    Returns:
        An argparse object with all the common arguments specified
        inside of the GAN framework
    """
    if not isinstance(parser, argparse.ArgumentParser):
        parser = argparse.ArgumentParser(script_initialization_example)

    available_optimizers = get_concrete_optimizers()

    # Discriminator arguments
    parser.add_argument("--d_lr",
                        type=float,
                        default=0.0001,
                        help="Learning rate assigned to the discriminator")
    parser.add_argument('--d_optim',
                        type=str,
                        default="Adam",
                        choices=available_optimizers,
                        help="Optimizer used for optimizing the discriminator")

    # Generator arguments
    parser.add_argument("--g_lr",
                        type=float,
                        default=0.0001,
                        help="Learning rate assigned to the generator")
    parser.add_argument('--g_optim',
                        type=str,
                        default="Adam",
                        choices=available_optimizers,
                        help="Optimizer used for optimizing the generator")

    def gan_argparse_inner(argparse_extension_func: t.Callable):
        @wraps(argparse_extension_func)
        def gan_argparse_exec(*args, **kwargs):
            # Run function for extending argument parser
            # The first argument is always the parser
            argparse_extension_func(parser, *args, **kwargs)
            return parser.parse_args()

        return gan_argparse_exec

    return gan_argparse_inner


def get_device_safe(gpu_index: int):
    """
    Given a gpu index, retrieve the target device if it is available.
    Raises RuntimeError if the
    Args:
        gpu_index: The index of gpu device to retrieve.
    Returns:
        The PyTorch device specified.
    """
    if gpu_index < 0:
        raise ValueError("Cannot specify negative index as gpu device index")
    if torch.cuda.is_available() and gpu_index - 1 <= torch.cuda.device_count():
        return torch.device(f"cuda:{gpu_index}")
    raise RuntimeError(f"The specified gpu at index: '{gpu_index}' is not available")


@gan_argparse("python3 app.py --d_lr 0.001 --g_lr 0.001")
def get_args(*args, **kwargs):
    parser, args = args

    # training Hyper-parameters
    parser.add_argument("--epochs",
                        type=int,
                        default=10,
                        help="The number of epochs to train model")
    parser.add_argument("--batch_size",
                        type=int,
                        default=16,
                        help="The size of mini-batch during training")
    # Note, we can also add support for multiple gpu training
    parser.add_argument("--gpu",
                        type=int,
                        default=0,
                        help="The target gpu to train on. ")

    # Enable to record stats
    parser.add_argument("--record_stats", dest="--record_stats", action="store_true")
    parser.add_argument("--disable_record_stats", dest="--record_stats", action="store_false")
    parser.set_defaults(record_stats=True)


# ACTUAL implementation logic begins here
# -------------------------------------------------------


class GeneratorMnist(nn.Module):
    def __init__(self, noise_vector_dim=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_vector_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 784),
            nn.Tanh()
        )
        init_weights(self)

    def forward(self, X) -> torch.Tensor:
        return self.model(X)


class DiscriminatorMnist(nn.Module):
    """
    Given a sample image, the goal is to determine whether
    the given example is real or fake.
    Since the Discriminator task is easier, the Generator
    should have slightly more modeling capacity to create
    an equilibrium which can be manipulated by increasing
    no. of parameters in the Generator or by regularizing the discriminator.
    """

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        init_weights(self)

    def forward(self, X) -> torch.Tensor:
        return self.model(X)


def train_model():
    # get arguments and hyper-parameters required for training model
    args: argparse.Namespace = get_args("test")

    # Hyper-parameters
    batch_size = args.batch_size
    disc_lr = args.d_lr
    gen_lr = args.g_lr
    noise_vector_dim = 100

    # Retrieve target device
    device = torch.device("cpu")
    if torch.cuda.is_available():
        get_device_safe(args.gpu)

    # Pre-define preprocessor logic
    data_transformer = transforms.Compose([
        # Convert PIL Images to Tensor.
        # this will also scale the Image values between 0 and 1.
        # Same as dividing all values by 255
        transforms.ToTensor(),

        # Because we are using the TanH function to output values, we need to
        # appropriately normalize the values such that GT images are scaled
        # between -1 and 1 so that we can fool the discriminator, since Generator
        # output will also be between -1 and 1.
        # if we use Sigmoid instead of TanH, the discriminator will easily win, since any
        # values outputted by generator that have values less than 0 are easily fake.
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Step 2. Load the dataset and create dataloader
    mnist_training = MNIST(train=True, download=True, root="data/", transform=data_transformer)
    dataloader = DataLoader(mnist_training, batch_size=batch_size, drop_last=True, shuffle=True)

    # Step 3. Initialize the model and optimizers
    disc = DiscriminatorMnist().to(device)
    gen = GeneratorMnist().to(device)
    disc_optimizer: torch.optim.Optimizer = get_optimizer(args.d_optim, lr=disc_lr, params=disc.parameters())
    gen_optimizer: torch.optim.Optimizer = get_optimizer(args.g_optim, lr=gen_lr, params=gen.parameters())

    binary_cross_entropy = nn.BCELoss()

    # Ground truth
    # We don't want our discriminator to get overconfident with its prediction
    # so we will make our real-labels 0.9
    # this is entirely optional!
    real_labels = torch.ones((batch_size, 1), device=device) - 0.1
    fake_labels = torch.zeros((batch_size, 1), device=device)

    disc.train()
    gen.train()

    # Step 4. Train the model
    for epoch in tqdm(range(args.epochs)):

        for i, (real_images, _) in enumerate(dataloader):
            # Initialize on the device if you use gpu instead of
            # doing .to(device), which initializes the tensor on the CPU and moves
            # it onto the GPU. This can create a significant bottleneck, as this operation
            # count grows linearly in proportion to size of the training set and epoch count
            noise_vector = torch.randn(batch_size, noise_vector_dim, device=device)
            # we need to flatten 28 x 28 image to 784 dim vector
            real_images = real_images.view(real_images.shape[0], -1)

            # Now, train generator
            # it is rewarded for tricking discriminator into thinking that the generated images are real
            gen_optimizer.zero_grad()
            fake_images = gen(noise_vector)
            disc_fake = disc(fake_images)
            loss_gen = binary_cross_entropy(disc_fake, real_labels)
            loss_gen.backward()
            gen_optimizer.step()

            # Train discriminator
            # Discriminator is rewarded for correctly classifying real and fake images.
            disc_optimizer.zero_grad()
            noise_vector = torch.randn(batch_size, noise_vector_dim, device=device)
            # Feed noise vector into the generator to get some fake images
            fake_images = gen(noise_vector)

            # Feed real images into disc
            disc_real = disc(real_images)
            disc_fake = disc(fake_images)

            # Calculate loss and back-prop
            # Discriminator is rewarded for correctly classifying real and fake images.
            loss_disc = (binary_cross_entropy(disc_real, real_labels)
                         + binary_cross_entropy(disc_fake, fake_labels)) / 2
            loss_disc.backward()
            disc_optimizer.step()

            # TODO: Later update this to make logging and visualization more pleasant
            if i % 100 == 0 and args.record_stats:
                print(f"[Epoch {epoch}, iter: {i}] - disc loss: {loss_disc}, gen loss: {loss_gen}")
                with torch.no_grad():
                    gen.eval()
                    noise_vector = torch.randn(batch_size, noise_vector_dim, device=device)
                    images_to_visualize = gen(noise_vector).view(real_images.shape[0], 1, 28, 28)
                    save_image(make_grid(images_to_visualize,
                                         nrow=4), f"mnist_epoch_{epoch}_iter_{i}_fake.jpg")
                    # save_image(make_grid(real_images.view(real_images.shape[0], 1, 28, 28),
                    #                      nrow=4), f"mnist_epoch_{epoch}_iter_{i}_real.jpg")
                    gen.train()


if __name__ == "__main__":
    train_model()
