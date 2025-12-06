import os
import glob


def load_papers(paper_dir="papers"):
    """
    Reads all text/markdown files in the papers directory and returns a combined context string.
    TODO: Add PDF parsing support using PyPDF2 or similar if libraries are available.
    """
    context = ""
    files = glob.glob(os.path.join(paper_dir, "*.txt")) + glob.glob(
        os.path.join(paper_dir, "*.md")
    )

    if not files:
        return "No external papers provided."

    context += "References from provided papers:\n"

    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                summary = content[:2000] + "..." if len(content) > 2000 else content
                context += (
                    f"\n--- Paper: {os.path.basename(file_path)} ---\n{summary}\n"
                )
        except Exception as e:
            print(f"Error reading paper {file_path}: {e}")

    return context


if __name__ == "__main__":
    print(load_papers())
