from huggingface_hub import snapshot_download

repo_ids = ["mistralai/Mistral-7B-v0.1",
            ]
# ["CausalLM/14B", "openchat/openchat_3.5", "THUDM/chatglm3-6b",]


for repo_id in repo_ids:
    local_dir = repo_id.split("/")[-1]
    cache_dir = local_dir + "/cache"
    snapshot_download(cache_dir=cache_dir,
                      local_dir=local_dir,
                      repo_id=repo_id,
                      local_dir_use_symlinks=False,
                      resume_download=True,
                      # allow_patterns=["*.model", "*.json", "*.bin",
                      #                 "*.py", "*.md", "*.txt", "*.safetensors",],
                      ignore_patterns=["*.msgpack",
                                      "*.h5", "*.ot", ],
                      endpoint="https://hf-mirror.com",
                      )