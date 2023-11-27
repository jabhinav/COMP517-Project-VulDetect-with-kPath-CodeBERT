import json


def pairwise_vul_detect(path_to_model1_results, path_to_model2_results):
	"""
	Detects pairwise vulnerability profile between two models.
	Our model uses max-pool to combine the logits from different paths. So the following logic that looks at pred
	from models that look at different code properties is applicable as the max pool will always pick which model says
	there is a vulnerability.
	"""
	print("Model 1 is %s" % path_to_model1_results)
	print("Model 2 is %s" % path_to_model2_results)
	
	model_1_results = json.load(open(path_to_model1_results, 'r'))
	model_2_results = json.load(open(path_to_model2_results, 'r'))
	
	# Sanity check: Keys should be the same
	assert list(model_1_results.keys()) == list(model_2_results.keys())
	
	total_samples = len(model_1_results)
	total_vul_samples = 0
	
	# Determine true positive for model 1
	model_1_true_positives = set()
	for key in model_1_results:
		
		if model_1_results[key]['label'] == 1:
			total_vul_samples += 1
		
		if model_1_results[key]['pred'] == 1 and model_1_results[key]['label'] == 1:
			model_1_true_positives.add(key)
			
	# Determine true positive for model 2
	model_2_true_positives = set()
	for key in model_2_results:
		if model_2_results[key]['pred'] == 1 and model_2_results[key]['label'] == 1:
			model_2_true_positives.add(key)
			
	# Identify sample where both models agree
	both_true_positives = model_1_true_positives.intersection(model_2_true_positives)
	
	# Identify sample where both models disagree
	model_1_only_true_positives = model_1_true_positives.difference(model_2_true_positives)
	model_2_only_true_positives = model_2_true_positives.difference(model_1_true_positives)
	
	# Show pairwise vulnerability profile (This is the same as Cross-Model Precision)
	print("Total vul samples: %d" % total_vul_samples)
	print(f"Model 1 TP: {len(model_1_true_positives)} ({len(model_1_true_positives) / total_vul_samples * 100:.2f}%)")
	print(f"Model 2 TP: {len(model_2_true_positives)} ({len(model_2_true_positives) / total_vul_samples * 100:.2f}%)")
	print(f"Agreements: {len(both_true_positives)} ({len(both_true_positives) / total_vul_samples * 100:.2f}%)")
	print(f"Model 1 only TP: {len(model_1_only_true_positives)} ({len(model_1_only_true_positives) / total_vul_samples * 100:.2f}%)")
	print(f"Model 2 only TP: {len(model_2_only_true_positives)} ({len(model_2_only_true_positives) / total_vul_samples * 100:.2f}%)")
	
	# ##################################################### MOE ##################################################### #
	# Now, let's assume a Mixture of Experts (MOE) model which predicts a vulnerability if either model predicts a
	# vulnerability, and non-vulnerability if both models predict non-vulnerability. This works because our model uses
	# max-pool to combine the logits from different paths.
	
	# [Mixture Of Experts] Identify total number of samples where either model is true positive
	moe_true_positives = model_1_only_true_positives.union(model_2_only_true_positives).union(both_true_positives)
	moe_true_negatives = set()
	for key in model_1_results:
		# If both models predict non-vulnerability, then it is a moe non-vulnerability
		if model_1_results[key]['pred'] == 0 and model_2_results[key]['pred'] == 0 and model_1_results[key]['label'] == 0:
			moe_true_negatives.add(key)
			
	print(f"MOE Precision: {len(moe_true_positives)} ({len(moe_true_positives) / total_vul_samples * 100:.2f}%)")
	print(f"MOE Accuracy: {(len(moe_true_negatives) + len(moe_true_positives)) / total_samples * 100:.2f}%")
	
	
	"""
	For example: Between CFG and PDG (trained with 2 paths and tested with 8 paths):
		Total vul samples: 1255
		Model 1 TP: 668 (53.23%)
		Model 2 TP: 552 (43.98%)
		Agreements: 463 (36.89%)
		Model 1 only TP: 205 (16.33%)
		Model 2 only TP: 89 (7.09%)
		MOE Precision: 757 (60.32%)
		MOE Accuracy: 63.98%
	"""
	
	
if __name__ == '__main__':
	pairwise_vul_detect(
		path_to_model1_results="./logging/results_CFG_k_train_2_k_test_8/results.json",
		path_to_model2_results="./logging/results_PDG_k_train_2_k_test_8/results.json"
	)
	