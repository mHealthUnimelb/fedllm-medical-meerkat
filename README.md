# FlowerTune LLM on Medical Dataset

This directory conducts federated instruction tuning with a pretrained [dmis-lab/meerkat-7b-v1.0](https://huggingface.co/dmis-lab/meerkat-7b-v1.0) model on a [Medical dataset](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards).
We use [Flower Datasets](https://flower.dev/docs/datasets/) to download, partition and preprocess the dataset.
Flower's Simulation Engine is used to simulate the LLM fine-tuning process in federated way,
which allows users to perform the training on a single GPU.

## PEFT Adapter

The fine-tuning results have been submitted as a PEFT adapter and can be accessed here:

- [FlowerTune-meerkat-7b-Instruct-Medical-PEFT](https://github.com/mHealthUnimelb/fedllm-medical-meerkat/tree/main/flowertune-eval-medical/peft_40)

## Methodology

This experiment performs federated LLM fine-tuning with [LoRA](https://arxiv.org/pdf/2106.09685) using the [🤗PEFT](https://huggingface.co/docs/peft/en/index) library.
The clients' models are aggregated with FedProx strategy.

### meerkat-7b-v1.0

For the **Meerkat-7b-v1.0 Instruct** model, we adopted the following fine-tuning methodology:

- **Precision**: bf16 for model weights, tf32 for gradients and optimizer states.
- **Quantization**: 4-bit quantization for reduced memory usage.
- **Optimizer**: Paged AdamW 8-bit for effective optimization under constrained resources.
- **LoRA Configuration**:
  - Rank (r): 8
  - Alpha: 32
  - Target Modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Training Configuration**:
  - Batch size: 16
  - Maximum number of steps: 6
  - Warmup steps: 2
  - Total number of rounds: 100
  - Fraction fit per round: 0.15
- **Learning Rate Scheduler**: Constant learning rate scheduler with warmup steps, where:
  - Maximum LR: 5e-5
  - Minimum LR: 1e-6
- **Strategy**: FedProx

When bf16 and tf32 are enabled, model weights are stored in bf16 format, while gradients are computed in half-precision and converted to full 32-bit precision for updates.

### Evaluation Results

- **pubmedqa**: 0.3600
- **medqa**: 0.1367
- **medmcqa**: 0.2577
- **careqa**: 0.1376
- **average**: 0.2230

### Communication Budget

3084.70 Megabytes

## Environments setup

Project dependencies are defined in `pyproject.toml`. Install them in an activated Python environment with:

```shell
pip install -e .
```

## Experimental setup

The dataset is divided into 20 partitions in an IID fashion, a partition is assigned to each ClientApp.
We randomly sample a fraction (0.15) of the total nodes to participate in each round, for a total of `100` rounds.
All settings are defined in `pyproject.toml`.

> [!IMPORTANT]
> Please note that `[tool.flwr.app.config.static]` and `options.num-supernodes` under `[tool.flwr.federations.local-simulation]` are not allowed to be modified for fair competition if you plan to participated in the [LLM leaderboard](https://flower.ai/benchmarks/llm-leaderboard).


## Running the challenge

Run the challenge with default config values.
The configs are defined in `[tool.flwr.app.config]` entry of `pyproject.toml`, and are loaded automatically.

```bash
flwr run
```

## Model saving

The global PEFT model checkpoints are saved every 5 rounds after aggregation on the sever side as default, which can be specified with `train.save-every-round` under [tool.flwr.app.config] entry in `pyproject.toml`.

> [!NOTE]
> Please provide the last PEFT checkpoint if you plan to participated in the [LLM leaderboard](https://flower.ai/benchmarks/llm-leaderboard).
