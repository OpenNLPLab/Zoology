from torch import nn

try:
    from hgru import SHgruV44Fast
except:
    SHgruV44Fast = None
    
class Hgrn2(nn.Module):
    def __init__(
        self,
        d_model,
        expand_ratio=2,
        act_fun="silu",
        uv_act_fun="sigmoid",
        use_norm=True,
        bias=True,
        norm_type="layernorm",
        **kwargs,
    ):
        super().__init__()
        
        self.mixer = SHgruV44Fast(
            embed_dim=d_model,
            expand_ratio=expand_ratio,
            act_fun=act_fun,
            uv_act_fun=uv_act_fun,
            use_norm=use_norm,
            bias=bias,
            norm_type=norm_type,
        )
        self.embed_dim = d_model
        self.expand_ratio = expand_ratio
        
    def forward(self, x):
        return self.mixer(x)
    
    def state_size(self, batch_size: int=1, sequence_length: int=2048):
        return 2 * self.embed_dim * self.expand_ratio