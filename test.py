from pathlib import Path

pdf_path = Path("documents/microsoft_10k.pdf")

print("File exists:", pdf_path.exists())
print("Full path:", pdf_path.resolve())

if pdf_path.exists():
    with open(pdf_path, "rb") as f:
        print("PDF header:", f.read(20))