import argparse
import typing as t
import inspect
from functools import wraps

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.mnist import MNIST


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

    # Discriminator arguments
    parser.add_argument("--d_lr",
                        type=float,
                        default=0.001,
                        help="Learning rate assigned to the discriminator")
    parser.add_argument('--d_optim',
                        type=str,
                        default="Adam",
                        help="Optimizer used for optimizing the discriminator")

    # Generator arguments
    parser.add_argument("--g_lr",
                        type=float,
                        default=0.001,
                        help="Learning rate assigned to the generator")
    parser.add_argument('--g_optim',
                        type=str,
                        default="Adam",
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
                        default=10,
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
    """

    """

    def __init__(self, noise_vector_dim=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_vector_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, X) -> torch.Tensor:
        return self.generator(X)


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
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, X) -> torch.Tensor:
        return self.model(X)


def train_model():
    # get arguments and hyper-parameters required for training model
    args: argparse.Namespace = get_args("test")

    # Hyperparameters
    batch_size = args.batch_size
    disc_lr = args.d_lr
    gen_lr = args.g_lr
    noise_vector_dim = 100

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
        transforms.Normalize((0.5, ), (0.5, ))
    ])

    # Step 2. Load the dataset and create dataloader
    mnist_training = MNIST(train=True, download=True, transform=data_transformer)
    dataloader = DataLoader(mnist_training, batch_size=batch_size, drop_last=True)

    # Step 3. Initialize the model and optimizers
    disc_optimizer: torch.optim.Optimizer = get_optimizer(args.d_optim, lr=disc_lr)
    gen_optimizer: torch.optim.Optimizer = get_optimizer(args.g_optim, lr=gen_lr)

    device = torch.device(f'cuda:{args.gpu}') if torch.cuda.is_available() and args.gpu - 1 <= torch.cuda.device_count() else

    # Step 4. Train the model
    for epoch in tqdm(20):
        for x_train, _ in tqdm(dataloader):
            random_noise = torch.randn(batch_size, noise_vector_dim)


if __name__ == "__main__":
    train_model()
