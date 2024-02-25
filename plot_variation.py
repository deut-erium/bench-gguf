import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Mistral-7B-Instruct-v0.2.csv")

plt.figure(figsize=(10, 6))

filtered_df = df[
         (df['model_filename']=='quantized/Mistral-7B-Instruct-v0.2/Mistral-7B-Instruct-v0.2-Q4_0.gguf') &
         (df['n_prompt']==512) &
         (df['n_gpu_layers']==0)
         ]

plt.errorbar(x=filtered_df['n_threads'], y=filtered_df['avg_ts'], yerr=filtered_df['stddev_ts'], linestyle='None', marker='o', label='Average tokens per second')
plt.xlabel('Number of threads')
plt.ylabel('Average Tokens per second')
