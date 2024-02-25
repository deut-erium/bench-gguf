import sys
import os
model_name = sys.argv[1]
model_owner, model_base_name = model_name.split("/")

cache_dir = f"~/.cache/huggingface/hub/models--{model_owner}--{model_base_name}"

for files in ["*.bin", "config.json",  "tokenizer.model"]:
    os.system(f'huggingface-cli download --local-dir models/{model_base_name} {model_name} --include "{files}"')

converted_path = f"models/{model_base_name}"

gguf_f32_model = f"{converted_path}/ggml-model-f32.gguf"

if not os.path.exists(gguf_f32_model):
    os.system(f"python3 convert.py {converted_path} --outtype f32")

if not os.path.exists("./quantized"):
    os.mkdir("quantized")

for quantization in ["Q4_0", "Q4_1", "Q5_0", "Q5_1", "IQ2_XXS", "IQ1_S", "Q2_K", "Q2_K_S", "IQ3_XXS", "Q3_K",
                      "Q3_K_XS", "Q3_K_S", "Q3_K_M", "Q3_K_L", "IQ4_NL", "Q4_K", "Q4_K_S", "Q4_K_M", "Q5_K",
                     "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0", "F16", "F32"]:
    model_dir_path = f"./quantized/{model_base_name}"
    if not os.path.exists(model_dir_path):
        os.mkdir(model_dir_path)
    quantized_name = f"./quantized/{model_base_name}/{model_base_name}-{quantization}.gguf"
    if not os.path.exists(quantized_name):
        os.system(f"./quantize {gguf_f32_model} {quantized_name} {quantization}")

# os.system(f"rm -rf {cache_dir} {converted_path}")
