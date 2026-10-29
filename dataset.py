
import torch

path = "data/cifar10_preactresnet18_badnet_0_01/attack_result.pt"

result = torch.load(
    path,
    map_location="cpu",
    weights_only=False
)

print("Available top-level keys:", list(result.keys()))

for name in ["bd_train", "bd_test"]:
    print(f"\n--- {name} ---")

    data = result[name]

    for key, value in data.items():
        if key not in ["bd_data_container"]:
            print(key, "=", value)

    container = data["bd_data_container"]
    data_dict = container.get("data_dict", {})

    print("Number of entries:", len(data_dict))

    for i in list(data_dict.keys())[:2]:
        print(f"Entry {i}:", data_dict[i])