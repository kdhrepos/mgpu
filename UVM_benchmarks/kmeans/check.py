import sys

def compare_files(file1, file2):
    try:
        with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
            lines1 = f1.readlines()
            lines2 = f2.readlines()
        
        if (len(lines1) != len(lines2)):
            print("Number of line is different")
            sys.exit(1)
        
        for i, line in enumerate(lines1):
            print(f"Content1: {lines1[i]}")
            print(f"Content2: {lines2[i]}")

            if (lines1[i] != lines2[i]):
                print("Different line!")
                sys.exit(1)

        print("Identical content")
    
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":

    file1 = "results/standard/1000000_centroids.txt"
    file2 = "results/standard/1000000_centroids.txt"
    
    compare_files(file1, file2)
