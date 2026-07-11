import warnings


def configure_warnings():
    warnings.filterwarnings(
        "ignore",
        message="You are using `torch.load` with `weights_only=False`",
    )

    warnings.filterwarnings(
        "ignore",
        message="std\\(\\): degrees of freedom is <= 0",
    )
    