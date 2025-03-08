import torch
from diffusers import StableDiffusionPipeline
import gc

# メモリをクリア
gc.collect()

# モデルをロード
print("標準のモデルをロード中...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,
    safety_checker=None,
    requires_safety_checker=False
)

# CPUに移動
pipe = pipe.to("cpu")

# 各コンポーネントを正しい方法で異なるデータ型に設定
if hasattr(pipe, "text_encoder"):
    pipe.text_encoder = pipe.text_encoder.to(dtype=torch.float32)  # float16ではなくfloat32を使用

if hasattr(pipe, "vae"):
    pipe.vae = pipe.vae.to(dtype=torch.float32)  # bfloat16はMacでサポートされていない可能性があるためfloat32を使用

# メモリ最適化
pipe.enable_attention_slicing()


# プロンプト設定
prompt = "A cute anime girl,high-quality, 全身,detailed, anime, masterpiece"
negative_prompt = "blurry, low quality, extra limbs, worst quality"

# 画像生成
print("画像生成中...")
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=25,
    guidance_scale=7.5,
    height=512,
    width=512
).images[0]

# 画像保存
image.save("generated_image.png")
print("画像を保存しました: generated_image.png")
