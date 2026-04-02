import os
import json

def main():
	callgraph_path = os.path.join(os.path.dirname(__file__), 'callgraph.json')
	if not os.path.exists(callgraph_path):
		print(f"File not found: {callgraph_path}")
		return

	with open(callgraph_path, 'r', encoding='utf-8') as f:
		data = json.load(f)

	node_types = set()
	node_files = set()
	for node in data.get('nodes', []):
		t = node.get('type')
		if t is not None:
			node_types.add(t)
		file = node.get('file')
		if file is not None:
			node_files.add(file)

	for t in sorted(node_types):
		print(t)

	# filter array: skip files containing any of these substrings
	filter_strings = [
		#'depends',
        #'usr/include',
        #'include/c++',
        #'_deps',
        #'x86_64-linux-gnu',
        #'fmt-src',
        # only to hide
        'include/df',
        'library',
        'plugins'

		# add more strings as needed
	]

	print("\nFiles:")
	count = 0
	for f in list(node_files):
		if any(s in f for s in filter_strings):
			continue
		print(f)
		count += 1
		if count >= 100:
			break

if __name__ == "__main__":
	main()
