import os

def main():
    # Execute with python3 (syscall)
    # Iterate over all files in the directory
    directory = "./testcodes/"
    with open("output.txt", "w") as f:
        for filename in os.listdir(directory):
            if filename.endswith(".py"):
                # Get output from the file (if error, print error)
                o = os.system(f"python {directory}{filename}")
                output = "Error" if o != 0 else "Success"
                print(f"Output for {filename}: {output}")
                f.write(f"Output for {filename}: {output}\n")

if __name__ == "__main__":
    main()
