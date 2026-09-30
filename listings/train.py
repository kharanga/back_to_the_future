import json
from pathlib import Path

from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from PIL import Image
from torch.utils.data import Dataset
from trl import SFTConfig, SFTTrainer

from listings import ADAPTER_DIR, PROJECT_ROOT, TRAIN_FILE
from listings.model import SEED, add_lora_to_language_layers, load_base_model

EPOCHS = 3
LEARNING_RATE = 2e-4
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
MAX_SEQ_LENGTH = 2048
WARMUP_STEPS = 10
CHECKPOINT_DIR = PROJECT_ROOT / "adapters" / "checkpoints"


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def open_photo(relative_path: str) -> Image.Image:
    return Image.open(PROJECT_ROOT / relative_path).convert("RGB")


def with_loaded_photos(block: dict) -> dict:
    if block["type"] == "image":
        return {"type": "image", "image": open_photo(block["image"])}
    return block


def trainable_conversation(example: dict) -> dict:
    return {
        "messages": [
            {"role": message["role"], "content": [with_loaded_photos(block) for block in message["content"]]}
            for message in example["messages"]
        ]
    }


class PhotosLoadedOnAccess(Dataset):
    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        return trainable_conversation(self.examples[index])


def training_config() -> SFTConfig:
    return SFTConfig(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        logging_steps=10,
        save_strategy="epoch",
        seed=SEED,
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=MAX_SEQ_LENGTH,
    )


def main():
    examples = PhotosLoadedOnAccess(read_jsonl(TRAIN_FILE))
    print(f"training examples: {len(examples)}")

    model, tokenizer = load_base_model()
    model = add_lora_to_language_layers(model)
    FastVisionModel.for_training(model)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=examples,
        args=training_config(),
    )
    trainer.train()

    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"adapter saved to {ADAPTER_DIR}")


if __name__ == "__main__":
    main()
