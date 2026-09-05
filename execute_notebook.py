# pyrefly: ignore [missing-import]
import nbformat
from nbclient import NotebookClient

def run_notebook():
    print("Loading iris_classification.ipynb...")
    with open("iris_classification.ipynb", "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    print("Executing notebook cells using nbclient...")
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()

    print("Saving executed notebook back to iris_classification.ipynb...")
    with open("iris_classification.ipynb", "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print("Notebook executed and updated successfully with zero errors!")

if __name__ == "__main__":
    run_notebook()
