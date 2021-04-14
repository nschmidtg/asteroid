from asteroid_filterbanks import make_enc_dec
from ..masknn.base import BaseUNet


class UNet(BaseUNet):
    """UNet as proposed in [1].

    Args:
        stft_n_filters (int) Number of filters for the STFT.
        stft_kernel_size (int): STFT frame length to use.
        stft_stride (int, optional): STFT hop length to use.
        sample_rate (float): Sampling rate of the model.

    References
        - [1] : "SINGING VOICE SEPARATION WITH DEEP U-NET
        CONVOLUTIONAL NETWORKS", Andreas Jansson et al.
        https://ejhumphrey.com/assets/pdf/jansson2017singing.pdf
    """

    def __init__(
        self, stft_n_filters=1024, stft_kernel_size=1024, stft_stride=768, sample_rate=8192.0
    ):
        self.stft_n_filters = stft_n_filters
        self.stft_kernel_size = stft_kernel_size
        self.stft_stride = stft_stride
        self.sample_rate = sample_rate

        encoder, decoder = make_enc_dec(
            "stft",
            n_filters=stft_n_filters,
            kernel_size=stft_kernel_size,
            stride=stft_stride,
            sample_rate=sample_rate,
        )
        super().__init__(encoder, decoder)

    def get_model_args(self):
        """Arguments needed to re-instantiate the model."""
        model_args = {
            "stft_n_filters": self.stft_n_filters,
            "stft_kernel_size": self.stft_kernel_size,
            "stft_stride": self.stft_stride,
            "sample_rate": self.sample_rate,
        }
        return model_args
