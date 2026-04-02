import json
import os

filter_strings = [
    'depends',
    'usr/include',
    'include/c++',
    '_deps',
    'x86_64-linux-gnu',
    'fmt-src',
    'proto'
]

def should_filter(file_path):
    return any(s in file_path for s in filter_strings)

def main():
    input_path = os.path.join(os.path.dirname(__file__), 'fileGraph.json')
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    nodes = data.get('nodes', [])
    filtered_nodes = [node for node in nodes if not should_filter(node.get('file', ''))]
    data['nodes'] = filtered_nodes

    output_path = input_path + '.cleaned'  # Overwrite original file, or change as needed
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

if __name__ == '__main__':
    main()
