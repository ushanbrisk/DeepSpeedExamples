import torch



file1 = '/ssd2/debug_20250617/step_4'

file2 = '/ssd2/debug_20250617/step_36'

file3 = '/ssd2/debug_20250617/step_76'

small_static_dict1 = torch.load(file1, map_location="cpu")

small_static_dict2 = torch.load(file2, map_location="cpu")

small_static_dict3 = torch.load(file3, map_location="cpu")

a =1