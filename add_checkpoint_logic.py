import json
with open("kaggle_train_libri4mix.ipynb", "r") as f:
    nb = json.load(f)
training_start_idx = -1
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "markdown" and any("Start Training" in line for line in cell["source"]):
        training_start_idx = i
        break
if training_start_idx != -1:
    markdown_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Resume from Checkpoint (Optional)\n",
            "If you have a previously saved training run in your Kaggle inputs, you can copy it to the working directory so SpeechBrain automatically resumes from it.\n",
            "Uncomment and modify the paths below to use it."
        ]
    }
    code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# CHECKPOINT_INPUT_DIR = \"/kaggle/input/YOUR-CHECKPOINT-DATASET/save\"\n",
            "# EXPERIMENT_SAVE_DIR = \"results/kaggle_libri4mix/1234/save\" # Ensure this matches your config output_folder/save\n",
            "\n",
            "# !mkdir -p {EXPERIMENT_SAVE_DIR}\n",
            "# !cp -r {CHECKPOINT_INPUT_DIR}/* {EXPERIMENT_SAVE_DIR}/\n",
            "# print(f\"Copied checkpoints to {EXPERIMENT_SAVE_DIR}. Training will automatically resume from the latest epoch.\")"
        ]
    }
    nb["cells"].insert(training_start_idx, markdown_cell)
    nb["cells"].insert(training_start_idx + 1, code_cell)
    with open("kaggle_train_libri4mix.ipynb", "w") as f:
        json.dump(nb, f, indent=1)
    print("Notebook patched.")
else:
    print("Could not find Start Training cell.")