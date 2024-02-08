from soundstream_net import Encoder, Decoder, ResidualVectorQuantizer
import torch
import torch.nn as nn
import librosa


def easy_solution_pad(x: torch.Tensor):
    """
    length of decoded audio is always in the form 270+n*320, where 320 is the compression factor.
    that is why it is needed to pad the audio before correctly
    """
    if type(x) != torch.Tensor:
        x = torch.Tensor(x)
    length = x.shape[-1]
    if length < 590:
        raise Exception
    pad_length = (length - 270) % 320
    if pad_length != 0:
        pad_length = 320 - pad_length
    #                              Pad first dimension
    x = torch.nn.functional.pad(x, (0, pad_length), "constant", 0)
    return x


class EncoderDecoder(nn.Module):
    def __init__(
            self,
            n_channels: int = 32,
            num_quantizers: int = 8,
            num_embeddings: int = 1024,
            padding: str = "same"
    ):
        super().__init__()
        self.encoder = Encoder(n_channels, padding)
        self.decoder = Decoder(n_channels, padding)
        self.quantizer = ResidualVectorQuantizer(
            num_quantizers, num_embeddings, n_channels * 16)

    def forward(self, x: torch.Tensor):
        x = easy_solution_pad(x)
        x = x.reshape(1, -1)
        y = self.encode(x)
        z = self.decode(y)
        assert x.shape[-1] == z.shape[-1]
        return z

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
    model = EncoderDecoder(**kwargs)
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


def test():
    model = soundstream_16khz(checkpoint=r"C:\\Users\\dudni\\Downloads\\soundstream_16khz-20230425.ckpt"
                              , pretrained=True)
    x, sr = librosa.load(r"C:\\Study\\Thesis\\Test_compression\\test.flac", sr=16000)
    y = model(x)

    return y


if __name__ == "__main__":
    test()