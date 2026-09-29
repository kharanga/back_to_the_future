from unsloth import FastVisionModel

from listings import ADAPTER_DIR

BASE_MODEL = "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit"
MAX_IMAGE_PIXELS = 768 * 768
LORA_RANK = 16
SEED = 3407


def limit_image_pixels(tokenizer):
    tokenizer.image_processor.max_pixels = MAX_IMAGE_PIXELS
    return tokenizer


def load_base_model():
    model, tokenizer = FastVisionModel.from_pretrained(
        BASE_MODEL,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    return model, limit_image_pixels(tokenizer)


def add_lora_to_language_layers(model):
    return FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=LORA_RANK,
        lora_alpha=LORA_RANK,
        lora_dropout=0,
        bias="none",
        random_state=SEED,
    )


def load_finetuned_model():
    model, tokenizer = FastVisionModel.from_pretrained(str(ADAPTER_DIR), load_in_4bit=True)
    FastVisionModel.for_inference(model)
    return model, limit_image_pixels(tokenizer)
