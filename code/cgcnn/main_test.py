import difflib

# Read lines from both python files
with open("main.py", "r") as f1, open("main_og.py", "r") as f2:
    file1_lines = f1.readlines()
    file2_lines = f2.readlines()

# Generate and print the unified diff
diff = difflib.unified_diff(
    file1_lines, 
    file2_lines, 
    fromfile="main.py", 
    tofile="main_og.py"
)

# Print out the results line by line
for line in diff:
    print(line, end="")