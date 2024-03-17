from zoology.config import TrainConfig, ModelConfig, DataConfig, ModuleConfig, FunctionConfig, LoggerConfig
from zoology.data.associative_recall import MQARConfig

import os

cache_dir = os.getenv("cache_dir", "cache_dir")

input_seq_len = 64

config = TrainConfig(
    max_epochs=20,
    data=DataConfig(
        cache_dir=cache_dir,
        train_configs=[MQARConfig(num_examples=10_000, vocab_size=128, input_seq_len=input_seq_len)],
        test_configs=[MQARConfig(num_examples=1_000, vocab_size=128, input_seq_len=input_seq_len)],
    ),
    model=ModelConfig(
        vocab_size=128,
        sequence_mixer=ModuleConfig(
            name="zoology.mixers.attention.MHA"
        )
    ),
    run_id=f"test",
    logger=LoggerConfig(
        project_name="lcsm",
        entity="doraemonzzz"
    )
)

configs = [config]