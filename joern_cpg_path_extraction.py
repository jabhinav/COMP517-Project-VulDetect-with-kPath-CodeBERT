import errno
import shutil
import os
import random
from typing import Dict, Tuple, List
from path_analysis import count_sample_path_freq
import joern_utils.cpg_generator as cpg_generator
import json
from tqdm import tqdm
from datetime import datetime
import logging
import networkx as nx
from utils.timeout import timeout, TimeoutError


# Setup logging
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join('./logging', current_time)
if not os.path.exists(log_dir):
	os.makedirs(log_dir)
logging.basicConfig(filename=os.path.join(log_dir, 'logs.txt'), filemode='w',
					format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
					datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


def read_data(ftype):
	file_path = f'./Devign/{ftype}.jsonl'
	
	samples = []
	with open(file_path) as f:
		for line in f:
			sample = json.loads(line.strip())
			samples.append(sample)
			
	return samples


def create_cpg_data(ftype='test'):
	joern_cli_dir = "/Users/apple/bin/joern/joern-cli/"
	# Path to store the CPG binary files
	cpg_path = "Devign/cpg/"
	# Path to store the C files
	cache_dir = "Devign/c_dir/"
	
	if not os.path.exists(cpg_path):
		os.mkdir(cpg_path)
	
	# Read the data
	samples = read_data(ftype)

	# Create CPG binary files
	# total_slices = len(samples) // slice_size
	for i in tqdm(range(0, len(samples)), desc="Creating CPG binary files", total=len(samples)):
		sample = samples[i]

		if not os.path.exists(cache_dir):
			os.mkdir(cache_dir)
		
		# Write functions to files in cache_dir
		f_idx = sample['idx']
		func = sample['func']
		
		# Write function to files in cache_dir
		with open(f"{cache_dir}{f_idx}.c", 'w') as f:
			f.write(func)
		
		# Create CPG binary file
		cpg_file = cpg_generator.joern_parse(joern_cli_dir, cache_dir, cpg_path, f"{f_idx}_cpg")
		
		# print(f"Dataset {i} to cpg.")
		
		# Clear the files in cache_dir
		shutil.rmtree(cache_dir)
		
		# Export the graphs
		graph_dir = os.path.join(cpg_path, f"{f_idx}_graphs")
		if os.path.exists(os.path.join(graph_dir, "export.json")):
			continue
		else:
			cpg_generator.joern_export(joern_cli_dir, os.path.join(cpg_path, cpg_file), graph_dir + "/")
			
		# Remove the cpg file
		os.remove(os.path.join(cpg_path, cpg_file))


def get_unique_code_path(code_lines, _path_code, _path, vertices):
	# Traverse the path code,
	# If vertex code matches with any other vertex code in the path, use code of instruction containing the vertex
	# Else use the vertex code
	_path_code_unique = []
	for i in range(len(_path_code)):
		# Check if the vertex code is already present in the path
		if _path_code[i] in _path_code_unique:
			# Find the line number of the vertex
			vertex_id = _path[i]
			line_number = vertices[vertex_id]['LINE_NUMBER']
			line_number_end = vertices[vertex_id]['LINE_NUMBER_END']
			# Find the code of the line
			unique_code = '\n'.join(code_lines[line_number - 1:line_number_end])
			unique_code = unique_code.strip()
			_path_code_unique.append(unique_code)
		else:
			_path_code_unique.append(_path_code[i])
			
	return _path_code_unique


def extract_paths(src_vertices, sink_vertices, vertices, edges, min_pool_size=5, max_tries=200, timeout_secs=10):
	"""
	Extract the shortest paths first and if insufficient, extract k other unique simple paths from the graph
	:param src_vertices: Source vertices corresponding to vertex label METHOD
	:param sink_vertices: Sink vertices corresponding to vertex label METHOD_RETURN
	:param vertices:
	:param edges:
	:param min_pool_size: This is the minimum size of pool of k simple paths we want to keep other than shortest
	:param max_tries: This is the maximum number of tries we give to nx.all_simple_paths with increasing cutoff
	:param timeout_secs: This is the overall timeout for extracting k simple paths
	:return:
	"""
	# Create a directed graph
	G = nx.DiGraph()
	
	# Add the vertices
	for vertex_id in vertices.keys():
		G.add_node(vertex_id)
	
	# Add the edges
	for edge_id in edges:
		edge = edges[edge_id]
		G.add_edge(edge[0], edge[1])
	
	all_shortest_paths = []
	
	# Reset the max_tries
	shortest_path_cutoffs = {}
	for src_vertex in src_vertices:
		for sink_vertex in sink_vertices:
			
			if nx.has_path(G, source=src_vertex, target=sink_vertex):
				# To extract all the shortest paths
				shortest_paths = nx.all_shortest_paths(G, source=src_vertex, target=sink_vertex)
				shortest_paths = list(shortest_paths)
				shortest_paths = [tuple(path) for path in shortest_paths]
				all_shortest_paths.extend(shortest_paths)
				
				shortest_path_cutoffs[(src_vertex, sink_vertex)] = len(shortest_paths[0])

			else:
				shortest_path_cutoffs[(src_vertex, sink_vertex)] = None
				logger.info(f"No path between {src_vertex} and {sink_vertex}")
	
	# Let's try to extract k simple paths
	k_simple_paths = []
	
	# Only extract k simple paths if there are any shortest paths
	if len(all_shortest_paths) > 0:
		unique_simple_paths_to_extract = min_pool_size
		# if len(all_shortest_paths) < min_pool_size:
		# 	unique_simple_paths_to_extract = min_pool_size - len(all_shortest_paths)
		
		try:
			@timeout(timeout_secs, os.strerror(errno.ETIMEDOUT))
			def extract_simple_paths(_max_num_paths):
				_max_tries = max_tries
				for _src_vertex in src_vertices:
					for _sink_vertex in sink_vertices:
						
						if shortest_path_cutoffs[(_src_vertex, _sink_vertex)] is None:
							continue
						
						cutoff = shortest_path_cutoffs[(_src_vertex, _sink_vertex)] + 1  # Cutoff for nx.all_simple_paths
						while len(k_simple_paths) < _max_num_paths and _max_tries > 0:
							_paths = list(nx.all_simple_paths(G, source=_src_vertex, target=_sink_vertex, cutoff=cutoff))
							_paths = set(tuple(path) for path in _paths)
							_paths = list(_paths)
							
							# _paths = [tuple(next(nx.all_simple_paths(G, source=src_vertex, target=sink_vertex, cutoff=cutoff)))]
							
							# Remove the paths which are already in all_shortest_paths
							_paths = [path for path in _paths if path not in all_shortest_paths]
							# Remove the paths which are already in k_simple_paths
							_paths = [path for path in _paths if path not in k_simple_paths]
							# Add the paths to k_simple_paths
							k_simple_paths.extend(_paths)
							
							_max_tries -= 1
							cutoff += 1
							
			extract_simple_paths(unique_simple_paths_to_extract)
		except TimeoutError:
			logger.warning(f"TimeoutError while extracting k simple paths. Returning the shortest paths only.")
		
	logger.info(f"Total shortest paths: ({len(all_shortest_paths)})")
	logger.info(f"Total simple paths: ({len(k_simple_paths)})")

	return {
		'shortest_paths': all_shortest_paths,
		'k_simple_paths': k_simple_paths,
		'total_paths': len(all_shortest_paths) + len(k_simple_paths)
	}


def extract_vertices(
		remove_vertex_labels: List[str],
		src_vertex_label: str,
		sink_vertex_label: str,
		standby_sink_vertex_label: str,
		is_CFG: bool,
		is_PDG: bool,
		edges: Dict[int, Tuple[int, int]],
		vertices: Dict[int, Dict],
):
	vertices_ooi = {}  # Extract the vertices of interest from the edges
	graph_src_vertices = set()
	graph_sink_vertices = set()
	graph_standby_sink_vertices = set()
	invalid_edges = []  # Edges for which we could not trace the vertex properties
	for edge_id in edges:
		
		valid_edge = True
		
		edge = edges[edge_id]
		
		# ############################## Get the source vertex ############################### #
		edge_src_vertex_id = edge[0]
		edge_src_vertex = vertices[edge_src_vertex_id]
		
		if edge_src_vertex['label'] not in remove_vertex_labels:
			
			# Check if the edge originates from a graph source vertex
			if edge_src_vertex['label'] == src_vertex_label:
				graph_src_vertices.add(edge_src_vertex_id)
				vertices_ooi[edge_src_vertex_id] = {
					'label': edge_src_vertex['label'],
					'CODE': edge_src_vertex['properties']['CODE']['@value'] if 'CODE' in edge_src_vertex[
						'properties'].keys() else '',
					'LINE_NUMBER': edge_src_vertex['properties']['LINE_NUMBER']['@value']['@value'] if 'LINE_NUMBER' in
																									   edge_src_vertex[
																										   'properties'].keys() else 0,
				}
				
				if 'LINE_NUMBER_END' in edge_src_vertex['properties'].keys():
					vertices_ooi[edge_src_vertex_id]['LINE_NUMBER_END'] = \
					edge_src_vertex['properties']['LINE_NUMBER_END']['@value']['@value']
				else:
					vertices_ooi[edge_src_vertex_id]['LINE_NUMBER_END'] = vertices_ooi[edge_src_vertex_id][
						'LINE_NUMBER']
			
			# Else add the vertex with filter
			else:
				try:
					if edge_src_vertex_id not in vertices_ooi:
						
						if is_CFG:
							vertices_ooi[edge_src_vertex_id] = {
								'label': edge_src_vertex['label'],
								'CODE': edge_src_vertex['properties']['CODE']['@value'],
							}
						elif is_PDG:
							vertices_ooi[edge_src_vertex_id] = {
								'label': edge_src_vertex['label'],
								'CODE': edge_src_vertex['properties']['CODE']['@value'],
								'COLUMN_NUMBER': edge_src_vertex['properties']['COLUMN_NUMBER']['@value']['@value'],
								'LINE_NUMBER': edge_src_vertex['properties']['LINE_NUMBER']['@value']['@value'],
							}
							if 'LINE_NUMBER_END' in edge_src_vertex['properties'].keys():
								vertices_ooi[edge_src_vertex_id]['LINE_NUMBER_END'] = \
								edge_src_vertex['properties']['LINE_NUMBER_END']['@value']['@value']
							else:
								vertices_ooi[edge_src_vertex_id]['LINE_NUMBER_END'] = vertices_ooi[edge_src_vertex_id][
									'LINE_NUMBER']
				
				except KeyError:
					# # This means the vertex is not a statement
					# logger.warning(f" Src Vertex {src_vertex} is not added to ddg_vertices")
					valid_edge = False
		else:
			valid_edge = False
		
		# ############################## Get the destination vertex ############################### #
		edge_dst_vertex_id = edge[1]
		edge_dst_vertex = vertices[edge_dst_vertex_id]
		
		# Add the destination vertex to the ddg_vertices
		if edge_dst_vertex['label'] not in remove_vertex_labels:
			if edge_dst_vertex['label'] == sink_vertex_label or edge_dst_vertex['label'] == standby_sink_vertex_label:
				
				if edge_dst_vertex['label'] == sink_vertex_label:
					graph_sink_vertices.add(edge_dst_vertex_id)
				
				elif edge_dst_vertex['label'] == standby_sink_vertex_label:
					graph_standby_sink_vertices.add(edge_dst_vertex_id)
				
				vertices_ooi[edge_dst_vertex_id] = {
					'label': edge_dst_vertex['label'],
					'CODE': edge_dst_vertex['properties']['CODE']['@value'] if 'CODE' in edge_dst_vertex[
						'properties'].keys() else '',
					'LINE_NUMBER': edge_dst_vertex['properties']['LINE_NUMBER']['@value']['@value'] if 'LINE_NUMBER' in
																									   edge_dst_vertex[
																										   'properties'].keys() else 0,
				}
				
				if 'LINE_NUMBER_END' in edge_dst_vertex['properties'].keys():
					vertices_ooi[edge_dst_vertex_id]['LINE_NUMBER_END'] = \
					edge_dst_vertex['properties']['LINE_NUMBER_END']['@value']['@value']
				else:
					vertices_ooi[edge_dst_vertex_id]['LINE_NUMBER_END'] = vertices_ooi[edge_dst_vertex_id][
						'LINE_NUMBER']
			
			else:
				try:
					if is_CFG:
						vertices_ooi[edge_dst_vertex_id] = {
							'label': edge_dst_vertex['label'],
							'CODE': edge_dst_vertex['properties']['CODE']['@value'],
						}
					elif is_PDG:
						vertices_ooi[edge_dst_vertex_id] = {
							'label': edge_dst_vertex['label'],
							'CODE': edge_dst_vertex['properties']['CODE']['@value'],
							'COLUMN_NUMBER': edge_dst_vertex['properties']['COLUMN_NUMBER']['@value']['@value'],
							'LINE_NUMBER': edge_dst_vertex['properties']['LINE_NUMBER']['@value']['@value'],
						}
						if 'LINE_NUMBER_END' in edge_dst_vertex['properties'].keys():
							vertices_ooi[edge_dst_vertex_id]['LINE_NUMBER_END'] = \
							edge_dst_vertex['properties']['LINE_NUMBER_END']['@value']['@value']
						else:
							vertices_ooi[edge_dst_vertex_id]['LINE_NUMBER_END'] = \
							edge_dst_vertex['properties']['LINE_NUMBER']['@value']['@value']
				
				except KeyError:
					# # This means the vertex is not a statement
					# logger.warning(f"Dst Vertex {dst_vertex} is not added to ddg_vertices")
					valid_edge = False
		else:
			valid_edge = False
		
		# #################### TO remove the edges with one or both end points missing ##################### #
		if not valid_edge:
			invalid_edges.append(edge_id)
			
	return graph_src_vertices, graph_sink_vertices, graph_standby_sink_vertices, vertices_ooi, invalid_edges


def parse_CPG(ftype='test', edge_labels=["CDG", "REACHING_DEF"]):
	"""
	Extract Graphs with desired property of interest from Code Property Graphs generated via Joern
	:param ftype: train/valid/test
	:param edge_labels: CFG/CDG/REACHING_DEF(DDG)
	:return:
	"""
	
	# Path to store the CPG binary files
	cpg_path = f"Devign/cpg_{ftype}/"
	
	# Read the data
	new_data_file = dict()

	samples = read_data(ftype)
	
	# # For debugging
	# samples = samples[:20]
	# samples = [samples[5]]
	
	# For specific analysis
	is_CFG = len(edge_labels) == 1 and 'CFG' in edge_labels
	is_PDG = len(edge_labels) == 2 and "CDG" in edge_labels and "REACHING_DEF" in edge_labels
	
	if is_CFG:
		src_vertex_label = 'METHOD'
		sink_vertex_label = 'METHOD_RETURN'
		standby_sink_vertex_label = 'METHOD_RETURN'
		remove_vertex_labels = ['METHOD_PARAMETER_OUT']
	elif is_PDG:
		src_vertex_label = 'METHOD'
		sink_vertex_label = 'RETURN'
		standby_sink_vertex_label = 'METHOD_RETURN'  # For Functions with no return statement
		remove_vertex_labels = ['METHOD_PARAMETER_OUT', 'IDENTIFIER']
	else:
		raise ValueError(f"Analysis for {edge_labels} not implemented!")

	unique_vertex_labels = set()
	path_stats = []
	
	for i in tqdm(range(0, len(samples)), desc=f"Extracting {edge_labels} from CPGs", total=len(samples)):
		
		f_idx, func = str(samples[i]['idx']), samples[i]['func']
		logger.info(f"************ Extracting {edge_labels} from CPGs: {f_idx} ************")

		graph_file = os.path.join(cpg_path, f"{f_idx}_graphs/export.json")
		
		# ############################################## Load the CPG ############################################## #
		# # Check - 1: If the graph file exists
		if not os.path.exists(graph_file):
			logger.warning(f"Graph file not found for {f_idx}")
			new_data_file[f_idx] = {
				'shortest_paths': [[func]],  # Add the function as the shortest path
				'k_simple_paths': [],
				'total_paths': 1
			}
			continue
		with open(graph_file, 'r') as f:
			cpg_graphs = json.load(f)
	
		# ############################################ Extract the edges ########################################### #
		# Extract the {DDG/CFG/CDG} - edges of type REACHING_DEF
		edges_ooi: Dict[int, Tuple[int, int]] = {}
		for edge in cpg_graphs['@value']['edges']:
			if edge['label'] in edge_labels:
				edges_ooi[edge['id']['@value']] = (edge['outV']['@value'], edge['inV']['@value'])
	
		logger.debug(f"{'_'.join(edge_labels)} edges: {len(edges_ooi)}")
		
		# Check - 2: If there are no edges of type {edge_label}
		if len(edges_ooi) == 0:
			logger.warning(f"No {'_'.join(edge_labels)} edges found for {f_idx}")
			new_data_file[f_idx] = {
				'shortest_paths': [[func]],  # Add the function as the shortest path
				'k_simple_paths': [],
				'total_paths': 1
			}
			continue
		
		# ######################################## Extract the vertices ############################################# #
		# Extract all the vertices from the graphs
		vertices_all = {}
		for vertex in cpg_graphs['@value']['vertices']:
			vertices_all[vertex['id']['@value']] = vertex
			unique_vertex_labels.add(vertex['label'])
		
		graph_srcs, graph_sinks, graph_standby_sinks, vertices_ooi, invalid_edges = extract_vertices(
			remove_vertex_labels, src_vertex_label, sink_vertex_label, standby_sink_vertex_label, is_CFG, is_PDG,
			edges_ooi, vertices_all
		)
		
		# Bookkeeping
		logger.info(f"Total edges: {len(edges_ooi)}")
		logger.info(f"Total vertices: {len(vertices_ooi)}")
		
		# TODO: Check performance if both sink and standby sink vertices are used
		if len(graph_sinks) == 0:
			graph_sinks = graph_standby_sinks
			logger.warning(f"No sink vertices found for {f_idx}. Using standby sink vertices")
		
		# ############################################# Filtering ################################################# #
		# # Filter 1: Remove the invalid edges
		for edge_id in invalid_edges:
			del edges_ooi[edge_id]
		logger.info(f"Total edges after removing invalid ones: {len(edges_ooi)}")
		
		# # Remove the edges which directly connect src and sink vertices
		for edge_id in list(edges_ooi.keys()):
			edge = edges_ooi[edge_id]
			if edge[0] in graph_srcs and edge[1] in graph_sinks:
				del edges_ooi[edge_id]
		logger.info(f"Total edges after removing edges which directly connect src and sink vertices: {len(edges_ooi)}")
		
		# # Filter 2: Remove the vertices which are not connected to any edge
		vertices_connected = set()
		for edge_id in edges_ooi:
			edge = edges_ooi[edge_id]
			vertices_connected.add(edge[0])
			vertices_connected.add(edge[1])
		for vertex_id in list(vertices_ooi.keys()):
			if vertex_id not in vertices_connected:
				del vertices_ooi[vertex_id]
		graph_srcs = graph_srcs.intersection(vertices_connected)  # Update the src_vertices
		graph_sinks = graph_sinks.intersection(vertices_connected)  # Update the sink_vertices
		logger.info(f"Total vertices after removing vertices which are not connected to any edge: {len(vertices_ooi)}")
		logger.info(f"Src/Sink Vertices: {graph_srcs}/{graph_sinks}")
		
		# Update vertices_ooi with the vertex codes from vertices_connected
		for vertex_id in vertices_ooi:
			if vertex_id not in vertices_connected:
				del vertices_ooi[vertex_id]
		
		# ############################################# Debugging ################################################# #
		
		# # [Debug] Print the DDG with  edges showing vertex codes
		# print(f"[Debug]\nDDG for sample {f_idx} with edges showing vertex codes:")
		# for edge_id in list(edges_ooi.keys()):
		# 	edge = edges_ooi[edge_id]
		# 	src_vertex_id = edge[0]
		# 	dst_vertex_id = edge[1]
		# 	src_vertex_code = vertices_ooi[src_vertex_id]['CODE']
		# 	dst_vertex_code = vertices_ooi[dst_vertex_id]['CODE']
		# 	src_vertex_label = vertices_ooi[src_vertex_id]['label']
		# 	dst_vertex_label = vertices_ooi[dst_vertex_id]['label']
		# 	# src_vertex_row = ddg_vertices[src_vertex_id]['LINE_NUMBER']
		# 	# dst_vertex_row = ddg_vertices[dst_vertex_id]['LINE_NUMBER']
		# 	print(f"({src_vertex_label}) {src_vertex_code} -> ({dst_vertex_label}) {dst_vertex_code}")
		
		
		# ######################################## Extract the paths ############################################# #
		try:
			assert len(graph_srcs) > 0
			assert len(graph_sinks) > 0
			
			new_data_file[f_idx] = extract_paths(graph_srcs, graph_sinks, vertices_ooi, edges_ooi)
			
			# Let's collect the code flow represented by the shortest paths first
			if is_CFG:
				new_data_file[f_idx]['shortest_paths'] = [
					[vertices_ooi[vertex]['CODE'] for vertex in path[1:-1]] for path in
					new_data_file[f_idx]['shortest_paths']
				]
				new_data_file[f_idx]['k_simple_paths'] = [
					[vertices_ooi[vertex]['CODE'] for vertex in path[1:-1]] for path in
					new_data_file[f_idx]['k_simple_paths']
				]
			elif is_PDG:
				code_lines = func.split('\n')
				
				shortest_paths_code = []
				for _path in new_data_file[f_idx]['shortest_paths']:
					_path_code = [vertices_ooi[vertex]['CODE'] for vertex in _path[1:]]
					shortest_paths_code.append(
						get_unique_code_path(code_lines, _path_code, _path[1:], vertices_ooi)
					)
					
				k_simple_paths_code = []
				for _path in new_data_file[f_idx]['k_simple_paths']:
					_path_code = [vertices_ooi[vertex]['CODE'] for vertex in _path[1:]]
					k_simple_paths_code.append(
						get_unique_code_path(code_lines, _path_code, _path[1:], vertices_ooi)
					)
					
				# UNIQUE CODE PATHS
				shortest_paths_code = [list(x) for x in set(tuple(x) for x in shortest_paths_code)]
				k_simple_paths_code = [list(x) for x in set(tuple(x) for x in k_simple_paths_code)]
					
				new_data_file[f_idx]['shortest_paths'] = shortest_paths_code
				new_data_file[f_idx]['k_simple_paths'] = k_simple_paths_code
				
		except AssertionError:
			logger.warning(f"Assertion Error for {f_idx}. Skipping this sample. src/sink vertices: {graph_srcs}/{graph_sinks}")
			new_data_file[f_idx] = {
				'shortest_paths': [[func]],  # Add the function as the shortest path
				'k_simple_paths': [],
				'total_paths': 1
			}
			
		if new_data_file[f_idx]['total_paths'] == 0:
			logger.warning(f"No paths found for {f_idx}. Adding the function as the shortest path")
			new_data_file[f_idx]['shortest_paths'] = [[func]]
			new_data_file[f_idx]['total_paths'] = 1
		
		# Update the path stats
		path_stats.append(new_data_file[f_idx]['total_paths'])
	
	
	# ################################################ Book-keeping ################################################ #
	print(f"Unique vertex labels found: {unique_vertex_labels}")
	
	with open(os.path.join(log_dir, f"{ftype}_{'_'.join(edge_labels)}_path_data.json"), 'w') as f:
		json.dump(new_data_file, f, indent=4)

	logger.info("************ Path stats ************")
	# Calculate the path stats - min, max, avg
	path_stats = sorted(path_stats)
	logger.info(f"Min: {min(path_stats)}")
	logger.info(f"Max: {max(path_stats)}")
	logger.info(f"Avg: {sum(path_stats) / len(path_stats)}")
	
	count_sample_path_freq(ftype, samples, new_data_file)


if __name__ == '__main__':
	# create_cpg_data()
	parse_CPG()
	

