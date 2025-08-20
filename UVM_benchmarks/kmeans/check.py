import sys

def compare_files(file1, file2):
    try:
        with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
            content1 = f1.read()
            content2 = f2.read()

        if content1 != content2:
            print(f"Error: Files '{file1}' and '{file2}' have different contents!")
            sys.exit(1)
        else:
            print(f"Files '{file1}' and '{file2}' are identical.")
    
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare.py <file1> <file2>")
        sys.exit(1)

    file1 = "results/standard/1000000_centroids"
    file2 = "results/standard/1000000_centroids_mgpu"
    
    compare_files(file1, file2)
