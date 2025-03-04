#!./bin/python3.12

# this code is used to extract semantic shift data of words in the input list (events and words for main analysis.txt)
# it was only tested using python3.12
# the use of this code is described in my Master's thesis
# references to the measures and methods used are also found in my thesis, but a brief description is offered below
# for anyone who might want to use this code right asap

# hweat score goes from around -.2 to .2, center is 0, with a higher number
# meaning the topic word is closer to the positive set

# semantic projection score goes around 0 to 1, center is .5, with a higher number
# meaning the topic word is closer to the positive set
# to do: add statistical significance (or not)

# VC  score goes from 0 to 1, with 0 -> no coherence (that is, total change) from t1 to t2, and 1 -> no change
# LNC score goes from 0 to 1, with 0 -> no coherence (that is, total change) from t1 to t2, and 1 -> no change
#  J  score goes from 0 to 1, with 0 -> no coherence (that is, total change) from t1 to t2, and 1 -> no change


import numpy as np
from scipy.spatial import distance
import math
import pickle
import sys
import csv
import os
import multiprocessing

def load_embeddings(decade, embeddings_path):
	vocab_file = f"{embeddings_path}/{decade}-vocab.pkl"
	weights_file = f"{embeddings_path}/{decade}-w.npy"
	
	with open(vocab_file, 'rb') as f:
		vocab_list = pickle.load(f)
	weights = np.load(weights_file)
	word_vectors = {word: weights[index] for index, word in enumerate(vocab_list)}
    
	return word_vectors
    
def cosine_similarity(vec1, vec2):
	return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def hweat(topic, attributeA, attributeB, decade):
	simtA = []
	simtB = []
	for item in attributeA:
		if not np.all(word_vectors[decade][item] == 0):
			simtA.append(1 - distance.cosine(word_vectors[decade][topic], word_vectors[decade][item]))
	for item in attributeB:
		if not np.all(word_vectors[decade][item] == 0):
			simtB.append(1 - distance.cosine(word_vectors[decade][topic], word_vectors[decade][item]))

	return np.mean(simtA) - np.mean(simtB)
	
def average_vector(wordlist, decade):
	# wordlist is a list of words to be used as the indexes of vectors of interest
	wordcounter = 0
	vectmp = None
	for word in wordlist:
		if not np.all(word_vectors[decade][word] == 0):
			vector = word_vectors[decade][word]
			if vectmp is None:
				vectmp = vector  # initialize vectmp with the first vector
			else:
				vectmp += vector
			wordcounter += 1

	avg_vector = vectmp / wordcounter  # calculate the average vector
	return avg_vector

def semantic_projection(topic, attributeA, attributeB, decade):
	if type(attributeA) == list:
		attributeA = np.array(average_vector(attributeA, decade))
	if type(attributeB) == list:
		attributeB = np.array(average_vector(attributeB, decade))
	AB = attributeB - attributeA
	AB_length = np.linalg.norm(AB)
	status = ''
	AB_unit = AB / AB_length
	AC = np.array(word_vectors[decade][topic]) - attributeA
	projection_length = np.dot(AC, AB_unit)
	projection = projection_length * AB_unit							# projection vector

	projection_point = attributeA + projection							# find the point of projection on the line
	if projection_length < 0:											# check if the projection falls outside the segment
		status = "projection falls before point A"
	elif projection_length > AB_length:
		status = "projection falls beyond point B"
	length_A_to_projection = np.linalg.norm(projection_point - attributeA)	# lengths of the segments
	length_projection_to_B = np.linalg.norm(attributeB - projection_point)	# lengths of the segments
	total_length = AB_length											# total length
	ratio_A_to_projection = length_A_to_projection / total_length		# ratio of the length of one segment to the total length
	ratio_projection_to_B = length_projection_to_B / total_length

	return projection_point, ratio_A_to_projection, ratio_projection_to_B, status
	
def vector_coherence(topic, decade1, decade2):
	return cosine_similarity(word_vectors[decade1][topic], word_vectors[decade2][topic])

def local_neighborhood_coherence(topic, decade1, decade2, neighbors_number = 25):	# neighbors = 25 because Hamilton et al. 2016b page 3
	topic_d1_embedding = word_vectors[decade1][topic]								# calculate the embeddings of the topic word in decades 1 and 2
	topic_d2_embedding = word_vectors[decade2][topic]
	distances_td1_d1 = {}
	distances_td1_d2 = {}
	for word, vec in word_vectors[decade1].items():
		if not np.all(vec == 0):
			distances_td1_d1[word] = distance.cosine(topic_d1_embedding, vec)		# distances between topic word embedded in decade 1 and every vector in decade 1
		else:
			distances_td1_d1[word] = np.float64(1)									# if the word is not embedded (meaning the vector is a series of 0) then set the distance to 1 (highest possible)
	for word, vec in word_vectors[decade2].items():
		if not np.all(vec == 0):
			distances_td1_d2[word] = distance.cosine(topic_d1_embedding, vec)		# distances between topic word embedded in decade 1 and every vector in decade 2
		else:
			distances_td1_d2[word] = np.float64(1)									# if the word is not embedded (meaning the vector is a series of 0) then set the distance to 1 (highest possible)

	closest_words_td1_d1 = sorted(distances_td1_d1, key = distances_td1_d1.get)[:neighbors_number + 1]
	closest_words_td1_d2 = sorted(distances_td1_d2, key = distances_td1_d2.get)[:neighbors_number + 1]

	neighbors_similarity_td1_d1 = []
	neighbors_similarity_td1_d2 = []
	neighbors_similarity_td2_d1 = []
	neighbors_similarity_td2_d2 = []
	for neighbor in closest_words_td1_d1[1:] + closest_words_td1_d2[1:]:
		if not np.all(word_vectors[decade1][neighbor] == 0):
			neighbors_similarity_td1_d1.append(1 - distance.cosine(topic_d1_embedding, word_vectors[decade1][neighbor]))
			neighbors_similarity_td2_d1.append(1 - distance.cosine(topic_d2_embedding, word_vectors[decade1][neighbor]))
		else:
			if high_precision == True:
				neighbors_similarity_td1_d1.append(1 - distance.cosine(topic_d1_embedding, average_vector(word_vectors[decade1], decade1)))
				neighbors_similarity_td2_d1.append(1 - distance.cosine(topic_d2_embedding, average_vector(word_vectors[decade1], decade1)))
		if not np.all(word_vectors[decade2][neighbor] == 0):
			neighbors_similarity_td1_d2.append(1 - distance.cosine(topic_d1_embedding, word_vectors[decade2][neighbor]))
			neighbors_similarity_td2_d2.append(1 - distance.cosine(topic_d2_embedding, word_vectors[decade2][neighbor]))
		else:
			if high_precision == True:
				neighbors_similarity_td1_d2.append(1 - distance.cosine(topic_d1_embedding, average_vector(word_vectors[decade2], decade2)))
				neighbors_similarity_td2_d2.append(1 - distance.cosine(topic_d2_embedding, average_vector(word_vectors[decade2], decade2)))
	
	return 1 - distance.cosine(neighbors_similarity_td1_d1 + neighbors_similarity_td1_d2, neighbors_similarity_td2_d1 + neighbors_similarity_td2_d2)

def jaccard(topic, decade1, decade2, neighbors_number = 25):
	topic_d1_embedding = word_vectors[decade1][topic]					# calculate the embeddings of the topic word in decades 1 and 2
	topic_d2_embedding = word_vectors[decade2][topic]
	distances_td1_d1 = {}
	distances_td1_d2 = {}
	for word, vec in word_vectors[decade1].items():
		if not np.all(vec == 0):
			distances_td1_d1[word] = distance.cosine(topic_d1_embedding, vec)		# distances between topic word embedded in decade 1 and every vector in decade 1
		else:
			distances_td1_d1[word] = np.float64(1)									# if the word is not embedded (meaning the vector is a series of 0) then set the distance to 1 (highest possible)
	for word, vec in word_vectors[decade2].items():
		if not np.all(vec == 0):
			distances_td1_d2[word] = distance.cosine(topic_d1_embedding, vec)		# distances between topic word embedded in decade 1 and every vector in decade 2
		else:
			distances_td1_d2[word] = np.float64(1)									# if the word is not embedded (meaning the vector is a series of 0) then set the distance to 1 (highest possible)
	closest_words_td1_d1 = sorted(distances_td1_d1, key = distances_td1_d1.get)[1:neighbors_number + 1]
	closest_words_td1_d2 = sorted(distances_td1_d2, key = distances_td1_d2.get)[1:neighbors_number + 1]
	
	return len(set(closest_words_td1_d1) & set(closest_words_td1_d2)) / len(set(closest_words_td1_d1) | set(closest_words_td1_d2))

def import_events(events_file):
	events = []
	with open(events_file, mode = 'r') as infile:
		reader = csv.DictReader(infile)
		for row in reader:
			if row and list(row.values())[0][0].isdigit():				# check if the first character of the first element is a digit
				events.append(row)
	return events

def compute_semantic_shift(events, index):
	output_filename_indexed = input_file_strip + "_processed_" + str(index) + ".csv"
	with open(output_filename_indexed, "w", newline = "") as output_file:
		output_data_fieldnames = ["year", "event", "positive", "human made", "human deaths", "distance from the US", "word", "word is relevant", "decade 1", "decade 2", "time frame is relevant", "VC", "LNC", "J"]
		writer = csv.DictWriter(output_file, output_data_fieldnames)
		writer.writeheader()
		tmprow = {}
		for event in events:												# for each row
			tmprow["event"] = event["event"]
			tmprow["year"] = event["year"]
			tmprow["positive"] = event["positive"]
			tmprow["human made"] = event["human made"]
			tmprow["human deaths"] = int(event["human deaths"])
			tmprow["distance from the US"] = int(event["distance from US"])
			for key, word in event.items():									# for each entry in the row
				if key.startswith("word"):									# for all entries in the row that should be words
					if word != None:										# for only the actual words in the row
						if "-" in event["year"]:							# save the decade in which the event happens (or starts)
							relevant_decade = int(event["year"].split('-')[0].strip()) - (int(event["year"].split('-')[0].strip()) % 10)
						else:
							relevant_decade = int(event["year"]) - (int(event["year"]) % 10)
							
						if word not in word_vectors[relevant_decade]:		# check if the word is embedded in the year in which the event happens, and if not move on to the next word
							print(f"The word {word} is not embedded in the decade of interest. Skipping word...")
							continue
						elif np.all(word_vectors[relevant_decade][word] == 0):
							print(f"The word {word} is not embedded in the decade of interest, even though there is an entry for it. Skipping word...")
							continue
						tmprow["word"] = word
						tmprow["word is relevant"] = True
						for decade in decades[:-1]:							# for all pairs of consecutive decades run the semantic shift measures
							if word not in word_vectors[decade]:
								print(f"The word {word} is not embedded in decade {decade}. Skipping decade pair...")
								continue
							elif np.all(word_vectors[decade][word] == 0):
								print(f"The word {word} is not embedded in decade {decade}, even though there is an entry for it. Skipping decade pair...")
								continue
							if  word not in word_vectors[decade + 10]:
								print(f"The word {word} is not embedded in decade {decade + 10}. Skipping decade pair...")
								continue
							elif np.all(word_vectors[decade + 10][word] == 0):
								print(f"The word {word} is not embedded in decade {decade + 10}, even though there is an entry for it. Skipping decade pair...")
								continue
							tmprow["decade 1"] = decade
							tmprow["decade 2"] = decade + 10
							tmprow["VC"] = vector_coherence(word, decade, decade + 10)
							tmprow["LNC"] = local_neighborhood_coherence(word, decade, decade + 10)
							tmprow["J"]= jaccard(word, decade, decade + 10)
							if "-" in event["year"]:
								if decade + 10 <= int(event["year"].split('-')[0].strip()) < decade + 20:
									tmprow["time frame is relevant"] = True
								else:
									tmprow["time frame is relevant"] = False
							else:
								if decade + 10 <= int(event["year"]) < decade + 20:
									tmprow["time frame is relevant"] = True
								else:
									tmprow["time frame is relevant"] = False
							writer.writerow(tmprow)

def multithreaded_compute_semantic_shift(event_data, chunk_size = 3):
	
	split_event_data = [event_data[i:i + chunk_size] for i in range(0, len(event_data), chunk_size)]	# define a list containing lists of chunk_size events each
	indexes = range(0, math.ceil(len(event_data) / chunk_size))											# define a list containing a number for each chunk, to use as index
	print(f"split_event_data len: {len(split_event_data)}, indexes len: {len(indexes)}")
	paired_lists = zip(split_event_data, indexes)
	with multiprocessing.Pool() as pool:
		pool.starmap(compute_semantic_shift, paired_lists)


if __name__ == "__main__":

	# check whether the right amount of arguments was provided, for the time being 2 to 3
	if len(sys.argv) < 2 or len(sys.argv) > 3:
		print(f"Usage: {sys.argv[0]} [input file] [high precision (0 = False, 1 = True), default 1]")
		sys.exit(1)
	elif len(sys.argv) == 3:
		high_precision = bool(int(sys.argv[2]))
	elif len(sys.argv) == 2:
		high_precision = True
	input_file = sys.argv[1]

	if high_precision == True:
		print(f"executing in high precision mode")
	elif high_precision == False:
		print(f"executing in low precision mode")

	input_file_strip = os.path.basename(input_file)
	input_file_strip, _ = os.path.splitext(input_file_strip)

	# loading the vectors
	decades = [1800, 1810, 1820, 1830, 1840, 1850, 1860, 1870, 1880, 1890, 1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990]
	embeddings_path = "/path/to/histwords/embeddings/eng-all_sgns/"
	word_vectors = {}

	for decade in decades:												# load all the embeddings
		print(f"loading {decade}'s embeddings...", end = "", flush = True)
		word_vectors[decade] = load_embeddings(decade, embeddings_path)
		print("\r", end = "", flush = True)
	print("Embeddings loaded successfully. Moving on...")

	events = import_events(input_file)
	multithreaded_compute_semantic_shift(events)

	# half of WEAT's implementation
	# set1 is from Greenwald, McGhee & Schwartz 1998 (IAT paper)
	# set2 is from Nosek, Banaji & Greenwald 2002

	positive_set1 =	["caress", "freedom", "health", "love", "peace", "cheer", "friend", "heaven", "loyal", "pleasure", "diamond", "gentle", "honest", "lucky", "rainbow", "diploma", "gift", "honor", "miracle", "sunrise", "family", "happy", "laughter", "paradise", "vacation"]
	negative_set1 =	["abuse", "crash", f"ilth", "murder", "sickness", "accident", "death", "grief", "poison", "stink", "assault", "disaster", "hatred", "pollute", "tragedy", "bomb", "divorce", "jail", "poverty", "ugly", "cancer", "evil", "kill", "rotten", "vomit", "agony", "prison"]
	positive_set2 =	["joy", "love", "peace", "wonderful", "pleasure", "friend", "laughter", "happy"]
	negative_set2 =	["agony", "terrible", "horrible", "nasty", "evil", "war", "awful", "failure"] # death has been removed at some point during the original experiment and substituted with horrible
	trust_set    = 	["trustworthy", "faith", "certainty", "assurance", "truth", "credible", "dependable", "authentic"] # some from Elo et al. 2014
	mistrust_set =	["suspicious", "illusion", "doubt", "discredit", "lie", "false", "fake", "dangerous", "biased"]
