from PIL import Image

from listings.parse import prompt_text

MAX_NEW_TOKENS = 40


def generation_messages(facts: dict) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": prompt_text(facts)},
            ],
        }
    ]


def generate_first_line(model, tokenizer, thumbnail: Image.Image, secondary: Image.Image, facts: dict) -> str:
    chat = tokenizer.apply_chat_template(generation_messages(facts), add_generation_prompt=True)
    inputs = tokenizer([thumbnail, secondary], chat, add_special_tokens=False, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, use_cache=True, do_sample=False)
    prompt_length = inputs["input_ids"].shape[1]
    return tokenizer.batch_decode(output[:, prompt_length:], skip_special_tokens=True)[0].strip()
