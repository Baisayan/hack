"""
data_processing.py
---------
This code contains certain useful helper functions that can be reused. 

USAGE:
python .\data_processing.py \
    --activation_dir C:\Users\SATI0004\Documents\SouthAI-Safety-hackathon\output\modified_advbench\mistral_12b_instruct\activations \
    --labels_csv C:\Users\SATI0004\Documents\SouthAI-Safety-hackathon\datasets\final_modified_advbench.csv \
    --output_filename combined_advbench_activations \
    --output_dir C:\Users\SATI0004\Documents\SouthAI-Safety-hackathon\output\modified_advbench\mistral_12b_instruct

NOTE: For output_filename, if you're running on the XSTest, please pass in the argument: 'combined_xstest_activations'.

"""

# Imports
import os
import torch
import argparse
import pandas as pd


def loading_indiv_activation(path, id):
    """
    Loads and resizes each individual activation vector.
    """
    print(f"INFO: Loading activations for {id}..")
    x = torch.load(path, map_location="cpu")

    # in case the activation vector is stored as a dict
    if isinstance(x, dict):
        print(f"INFO: Keys in activation file:", x.keys())
        x = x["activations"]

    x = x.detach().cpu().float().squeeze()

    # Standardize to [num_layers, hidden_dim]
    if x.ndim == 1:
        # [hidden_dim] -> [1, hidden_dim]
        x = x.unsqueeze(0)

    elif x.ndim == 2:
        # already in the correct format [num_layers, hidden_dim]
        pass
    elif x.ndim == 3:
        # [num_layers, seq_len, hidden_dim], takes the final token
        x = x[:, -1, :]

    elif x.ndim == 4:
        # [batch, num_layers, seq_len, hidden_dim]
        x = x[0, :, -1, :]
    else:
        raise ValueError(f"ERROR: Unexpected activation shape: {x.shape}")
    return x


def build_probe_dataset(labels_csv, activation_dir, output_dir, filename):
    """
    Merges all the individual activation vectors into one unified dataset for training the linear probe.
    """
    df = pd.read_csv(labels_csv)

    ids, labels, activations = [], [], []
    shape_per_activation = None

    label_map = {"safe": 0, "unsafe": 1}

    for _, row in df.iterrows():
        # converting the label into 0 or 1 
        id = str(row['id'])
        text_label = str(row['label']).strip().lower()
        label = label_map[text_label]

        # path to the activation.pt
        activation_path = os.path.join(activation_dir, f"{id}.pt")

        # loading the activation vector path
        if not os.path.exists(activation_path):
            print(f"ERROR: Missing activation file: {activation_path}")
            continue
        print(f"INFO: Loading all activations one-by-one...")
        x = loading_indiv_activation(activation_path, id)
        print(f"INFO: Successfully Loaded all activations! ")
        shape_per_activation = x.shape

        ids.append(id)
        labels.append(label)
        activations.append(x)

    if len(activations) == 0:
        raise RuntimeError(f"ERROR: No activation files were successfully loaded!")
    elif len(activations) != len(df):
        raise RuntimeError(f"ERROR: Not all activation files were properly merged.\nlen(activations) = {len(activations)}\nlen(df) = {len(df)}")
    
    # Stack all these into final tensor: [num_examples, num_layers, hidden_dim]
    X = torch.stack(activations, dim=0)

    # Labels: [num_examples]
    y = torch.tensor(labels, dtype=torch.long)

    # Creating the probe dataset
    probe_dataset = {
        "ids": ids,
        "X": X,
        "y": y,
        "label_map": label_map,
        "num_examples": len(ids),
        "activation_shape": tuple(shape_per_activation)
    }

    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    print(f"INFO: Saving the compiled dataset...")
    # saving the probe_dataset.pt
    output_file = os.path.join(output_dir, f"{filename}.pt")
    torch.save(probe_dataset, output_file)

    print(f"INFO: Successfully saved probe dataset to {output_file}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Number of examples: {len(ids)}")

# arguments
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--activation_dir", 
        type=str,
        required= True,
        help="Local Path to the directory where the individual activations are stored"
    )
    parser.add_argument(
        "--labels_csv", 
        type=str,
        required= True,
        help="Local FilePath of the dataset's CSV."
    )
    parser.add_argument(
        "--output_filename", 
        type=str,
        required= True,
        help="Output filename that you would want to save the compiled probe dataset."
    )
    parser.add_argument(
        "--output_dir", 
        type=str,
        required= True,
        help="Local path to the directory where you want to store the output."
    )
    
    # parsing and assigning the arguments
    args = parser.parse_args()

    labels_csv = args.labels_csv
    activation_dir = args.activation_dir
    output_dir = args.output_dir
    filename = args.output_filename

    build_probe_dataset(labels_csv, activation_dir, output_dir, filename)

if __name__ == "__main__":
    main()












