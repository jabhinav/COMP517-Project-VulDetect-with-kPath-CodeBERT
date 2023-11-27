import json
from collections import defaultdict


def read_data(ftype):
	file_path = f'./Devign/{ftype}.jsonl'
	
	samples = []
	with open(file_path) as f:
		for line in f:
			sample = json.loads(line.strip())
			samples.append(sample)
	
	return samples


def read_path_data(ftype):
	path_file = f'./Devign/{ftype}_CFG_path_data.json'
	with open(path_file, 'r') as f:
		data = json.load(f)
	return data


def count_sample_path_freq(ftype='valid', samples=None, path_data=None):
	
	samples = read_data(ftype) if samples is None else samples
	path_data = read_path_data(ftype) if path_data is None else path_data
	
	vuln_path_freq = defaultdict(int)
	benign_path_freq = defaultdict(int)
	
	print("[DEBUG] Ignoring paths of length 1.")
	for sample in samples:
		try:
			paths = path_data[str(sample['idx'])]
		except KeyError:
			continue
		
		total_paths = []
		
		for path in paths['shortest_paths']:
			if len(path) > 1:
				total_paths.append(path)
		
		for path in paths['k_simple_paths']:
			if len(path) > 1:
				total_paths.append(path)
				
		if sample['target'] == 1:
			vuln_path_freq[len(total_paths)] += 1
		else:
			benign_path_freq[len(total_paths)] += 1
	
	# Sort the path frequencies in ascending order
	vuln_path_freq = dict(sorted(vuln_path_freq.items(), key=lambda item: item[0]))
	benign_path_freq = dict(sorted(benign_path_freq.items(), key=lambda item: item[0]))
	
	print("Vuln Fn #Path to #Sample Freq")
	print(vuln_path_freq)
	print("Benign Fn #Path to #Sample Freq")
	print(benign_path_freq)
	
	print("Num Vuln Fns: ", sum(vuln_path_freq.values()))
	print("Vul Percent: ", sum(vuln_path_freq.values()) / (sum(vuln_path_freq.values()) + sum(benign_path_freq.values())))
	print("Num Benign Fns: ", sum(benign_path_freq.values()))
	print("Benign Percent: ", sum(benign_path_freq.values()) / (sum(vuln_path_freq.values()) + sum(benign_path_freq.values())))
	

if __name__ == '__main__':
	count_sample_path_freq()
	
	