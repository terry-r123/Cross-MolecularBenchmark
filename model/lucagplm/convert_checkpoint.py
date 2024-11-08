import torch


para_file_path = "/home/bingxing2/ailab/group/ai4bio/public/multi-omics/lucaone/llm/models/lucagplm/v2.0/token_level,span_level,seq_level,structure_level/lucaone_gplm/20231125113045/checkpoint-step5600000/pytorch.pth"
state_dict = torch.load(para_file_path, map_location=torch.device('cpu'))

new_state_dict = {}
for param_name, param_tensor in state_dict.items():
    param_name = "lucaone." + param_name
    print(f"Parameter Name: {param_name}")
    new_state_dict[param_name] = param_tensor
torch.save(new_state_dict, "/home/bingxing2/ailab/group/ai4bio/public/multi-omics/lucaone/llm/models/lucagplm/v2.0/token_level,span_level,seq_level,structure_level/lucaone_gplm/20231125113045/checkpoint-step5600000/pytorch_model.bin")