import torch

# 示例 diff 张量
diff = torch.tensor(
    [[29.1929, 34.0746],
        [33.7251, 42.0175]], device='cuda:0')

print(diff[0])
a = torch.sum((diff - 100).clip(0, 500), dim=1)
print(a)


b = torch.sum(diff, dim=1)
print(b)


c = torch.sum(diff, dim=0)
print(c)

# # 计算范数
# norm_values = torch.norm(diff, dim=1)
#
# # 计算最终结果
# result = torch.exp(-2 * norm_values)
#
# result2 = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
#
# # 打印结果
# print("Norm values:")
# print(norm_values)
# print("Exp(-2 * Norm values):")
# print(result)
# print(result2)
