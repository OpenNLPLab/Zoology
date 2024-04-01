import uuid
import numpy as np
from zoology.config import TrainConfig, ModelConfig, DataConfig, DataSegmentConfig, LoggerConfig
from zoology.data.associative_recall import MQARConfig
import os

sweep_id = uuid.uuid4().hex[:6]
sweep_name = "figure2" + sweep_id

cache_dir = os.getenv("cache_dir", "cache_dir")

VOCAB_SIZE = 8_192


configs = []
for d_model in [
    32,
    64, 
    128, 
    256, 
    # 512
]:
    for input_seq_len, num_kv_pairs in [
        (32, 2),
        (64, 4),
        (128, 8),
        (256, 16),
        # (512, 64),
    ]:
        if input_seq_len == 1024:
            batch_size = 64
        elif input_seq_len == 512:
            batch_size = 64
        elif input_seq_len == 256:
            batch_size = 256
        else:
            batch_size = 512


        factory_kwargs = {
            "num_kv_pairs": num_kv_pairs,
            "train_power_a": 0.01,
            "test_power_a": 0.01,
            "random_non_queries": False
        }

        data = DataConfig(
            train_configs=[MQARConfig(num_examples=100_000, vocab_size=VOCAB_SIZE, input_seq_len=input_seq_len, **factory_kwargs)],
            test_configs=[MQARConfig(num_examples=3_000, vocab_size=VOCAB_SIZE, input_seq_len=input_seq_len, **factory_kwargs)],
            batch_size=batch_size,
            cache_dir=cache_dir,
        )


        # for lr in  np.logspace(-4, -2, 4):
        for lr in [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]:
            MIXERS = {
                "lcsm": dict(
                    name="zoology.mixers.lcsm.Lcsm",
                    kwargs={
                        "dropout": 0.1,
                        "expand_dim": 128,
                        "c_type": 1,
                        "e_type": 1,
                        "o_type": 1,
                        "s_type": 1,
                        "o_learned": True,
                        "tau": 16,
                        "use_tau": True,
                        "t_type": 0,
                    },
                ),
            }
            
            for t_type in range(1, 4):
                for sequence_mixer in [
                    "lcsm",
                ]:
                    MIXERS[sequence_mixer]["kwargs"]["t_type"] = t_type
                    block_type = "TransformerBlock"

                    model = ModelConfig(
                        d_model=d_model,
                        n_layers=2,
                        block_type=block_type,
                        max_position_embeddings=input_seq_len if sequence_mixer == "attention" else 0,
                        vocab_size=VOCAB_SIZE,
                        sequence_mixer=MIXERS[sequence_mixer],
                        state_mixer=dict(name="torch.nn.Identity", kwargs={})
                    )
                    kwargs = MIXERS[sequence_mixer]["kwargs"]
                    config = TrainConfig(
                        model=model,
                        data=data,
                        learning_rate=lr,
                        max_epochs=64,
                        run_id=f"{sequence_mixer}-c{kwargs['c_type']}-e{kwargs['e_type']}-f{kwargs['o_type']}-s{kwargs['s_type']}-fl{kwargs['o_learned']}-tau{kwargs['tau']}-ut{kwargs['use_tau']}-t{kwargs['t_type']}-seqlen{input_seq_len}-dmodel{d_model}-lr{lr}-kv{num_kv_pairs}",
                        logger=LoggerConfig(
                            project_name="lcsm",
                            entity="doraemonzzz"
                        )

                    )
                    configs.append(config)