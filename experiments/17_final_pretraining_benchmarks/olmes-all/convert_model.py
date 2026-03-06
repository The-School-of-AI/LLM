import torch
from safetensors.torch import save_file


def inspect_checkpoint(checkpoint_file):
    ckpt = torch.load(
        checkpoint_file,
        map_location="cpu"
    )
    print(ckpt.keys())
    print(ckpt["global_steps"])


def extract_model():
    ckpt = torch.load(
        "checkpoint/mp_rank_00_model_states.pt",
        map_location="cpu",
        weights_only=True
    )

    state_dict = ckpt["module"]
    print("Number of parameters:", len(state_dict))

    torch.save(state_dict, "pytorch_model.bin")
    print("Saved pytorch_model.bin")


def convert_to_safetensors():
    state = torch.load("pytorch_model.bin", map_location="cpu", weights_only=True)

    # break shared memory references
    clean_state = {k: v.clone() for k, v in state.items()}

    save_file(clean_state, "model.safetensors")

    print("Saved model.safetensors successfully")


if __name__ == "__main__":
    checkpoint_file = "checkpoint/mp_rank_00_model_states.pt"
    inspect_checkpoint(checkpoint_file)