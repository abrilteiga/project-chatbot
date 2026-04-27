
def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def chunk_text(text, chunk_size = 300, overlap = 50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks