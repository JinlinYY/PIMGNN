# 测试数据样本
test_samples = [
    {
        "description": "水-乙醇-丙醇系统 (298K)",
        "smiles1": "O",
        "smiles2": "CCO",
        "smiles3": "CCCO",
        "temperature": 298.15
    },
    {
        "description": "水-甲醇-乙醇系统 (313K)",
        "smiles1": "O",
        "smiles2": "CO",
        "smiles3": "CCO",
        "temperature": 313.15
    },
    {
        "description": "水-乙醇-异丙醇系统 (333K)",
        "smiles1": "O",
        "smiles2": "CCO",
        "smiles3": "CC(C)O",
        "temperature": 333.15
    },
    {
        "description": "水-乙醇-丁醇系统 (298K)",
        "smiles1": "O",
        "smiles2": "CCO",
        "smiles3": "CCCCO",
        "temperature": 298.15
    },
    {
        "description": "乙醇-丙醇-丁醇系统 (308K)",
        "smiles1": "CCO",
        "smiles2": "CCCO",
        "smiles3": "CCCCO",
        "temperature": 308.15
    },
    {
        "description": "水-乙醇-乙酸乙酯系统 (303K)",
        "smiles1": "O",
        "smiles2": "CCO",
        "smiles3": "CCOC(C)=O",
        "temperature": 303.15
    },
    {
        "description": "水-甲醇-乙酸甲酯系统 (293K)",
        "smiles1": "O",
        "smiles2": "CO",
        "smiles3": "COC(C)=O",
        "temperature": 293.15
    },
    {
        "description": "水-乙醇-丙酮系统 (323K)",
        "smiles1": "O",
        "smiles2": "CCO",
        "smiles3": "CC(C)=O",
        "temperature": 323.15
    }
]

# 打印测试数据
print("=== LLE 预测测试数据样本 ===")
print("可以直接复制到前端界面进行测试")
print()

for i, sample in enumerate(test_samples, 1):
    print(f"测试样例 {i}: {sample['description']}")
    print(f"组分1 SMILES: {sample['smiles1']}")
    print(f"组分2 SMILES: {sample['smiles2']}")
    print(f"组分3 SMILES: {sample['smiles3']}")
    print(f"温度: {sample['temperature']} K")
    print("-" * 50)