from soundstream_net import Encoder, Decoder, ResidualVectorQuantizer
import torch
import torch.nn as nn


class EncoderDecoder(nn.Module):
    def __init__(
            self,
            n_channels: int = 32,
            num_quantizers: int = 8,
            num_embeddings: int = 1024,
            padding: str = "valid"
    ):
        super().__init__()
        self.encoder = Encoder(n_channels, padding)
        self.decoder = Decoder(n_channels, padding)
        self.quantizer = ResidualVectorQuantizer(
            num_quantizers, num_embeddings, n_channels * 16)

    def forward(self, x):
        return self.encode(x)

    def encode(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 2
        x = torch.unsqueeze(input, 1)
        x = self.encoder(x)
        x = torch.transpose(x, -1, -2)
        _, codes, _ = self.quantizer(x)
        return codes

    def decode(self, input: torch.Tensor) -> torch.Tensor:
        # input: [batch_size, length, num_quantizers]
        x = self.quantizer.dequantize(input)
        x = torch.transpose(x, -1, -2)
        x = self.decoder(x)
        x = torch.squeeze(x, 1)
        return x


def soundstream_16khz(checkpoint=None, pretrained=True, **kwargs):
    """SoundStream encoder decoder
    for loading from a checkpoint
    pretrained (bool): kwargs, load pretrained weights into the model
    """
    # Call the model, load pretrained weights
    model = EncoderDecoder()
    if not pretrained:
        model.eval()
        return model
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    if not checkpoint:
        checkpoint = '/home/woody/iwi1/iwi1010h/checkpoints/SoundStream/soundstream_16khz-20230425.ckpt'
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict['state_dict'], strict=False)
    model.eval()
    return model
