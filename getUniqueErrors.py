# open build_callgraph.log and parse it to get all unique lines
# print the unique lines

def get_unique_errors(log_file):
    unique_errors = set()
    with open(log_file, 'r') as f:
        for line in f:
            unique_errors.add(line.strip())
    return unique_errors

if __name__ == "__main__":
    log_file = 'build_callgraph.log'
    unique_errors = get_unique_errors(log_file)
    for error in unique_errors:
        print(error)