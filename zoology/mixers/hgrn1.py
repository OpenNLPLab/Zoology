from torch import nn

try:
    from hgru import Hgru1d
except:
    Hgru1d = None
    
class Hgrn1(nn.Module):
    def __init__(
        self,
        d_model,
        act_fun="silu",
        causal=True,
        use_triton=False,
        bias=True,
        **kwargs,
    ):
        super().__init__()
        
        self.mixer = Hgru1d(
            embed_dim=d_model,
            act_fun=act_fun,
            causal=causal,
            use_triton=use_triton,
            bias=bias,
        )
        self.embed_dim = d_model
        
    def forward(self, x):
        return self.mixer(x)
    
    def state_size(self, batch_size: int=1, sequence_length: int=2048):
        return 2 * self.embed_dim