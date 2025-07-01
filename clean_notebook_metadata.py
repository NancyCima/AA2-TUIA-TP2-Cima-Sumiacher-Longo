import nbformat

# Input and output notebook paths
input_path = "Problema1-Audio MNIST/TP2_AAII.ipynb"
output_path = "Problema1-Audio MNIST/TP2_AAII.ipynb"

# Load the notebook
with open(input_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# Remove global metadata
nb.metadata = {}

# Remove metadata from each cell
for cell in nb.cells:
    cell.metadata = {}

# Save the cleaned notebook
with open(output_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Notebook cleaned and saved to {output_path}")
