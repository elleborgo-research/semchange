#!/usr/bin/env python


# this code receives a text file or wikipedia article url and returns the num_words most frequent words in its body, excluding those in the stop_words list

from collections import Counter
import requests
from bs4 import BeautifulSoup
import re
import sys

def most_common_words(file_path, exclude_words, num_words=20):
	if exclude_words is None:
		exclude_words = []

	# check whether a url or a file path was provided as argument
	if any(x in file_path for x in ['www.', 'http://', 'https://', '.com', '.org', '.net']):
		isfile = False
	else:
		isfile = True

	if isfile == False:
		response = requests.get(file_path)
		if response.status_code == 200:
			# Parse the HTML content
			soup = BeautifulSoup(response.text, 'html.parser')
			# Remove unwanted sections
			for section in soup.find_all(['div', 'span'], class_=['toc', 'reference', 'references', 'reflist', 'further_reading', 'external_links']):
				section.decompose()
            # Find the main content
			content_div = soup.find('div', {'id': 'mw-content-text'})
			if content_div:
				# Get all paragraphs after the first heading
				main_content = []
				
				for element in content_div.find_all(['p', 'h2']):
					# Stop at References or Further reading sections
					if element.name == 'h2':
						heading_text = element.get_text().lower()
						if any(x in heading_text for x in ['references', 'further reading', 'see also', 'external links']):
							break

					main_content.append(element.get_text())

				text = ' '.join(main_content)
			else:
				text = soup.get_text()
		else:
			print(f"Failed to retrieve the page. Status code: {response.status_code}")
			return []
	else:
		with open(file_path, 'r', encoding='utf-8') as file:
			text = file.read()

	
	words = re.findall(r'\b\w+\b', text.lower())							# remove punctuation and convert to lowercase
	filtered_words = [word for word in words if word not in exclude_words]	# remove stop words
	word_counts = Counter(filtered_words)
	common_words = word_counts.most_common(num_words)
	
	return common_words

if len(sys.argv) < 2 or len(sys.argv) > 3:
	print(f"Usage: {sys.argv[0]} input_file_or_url [number of words to display, default 20]\nSee the code for the excluded function words")
	sys.exit(1)

stop_words = ['the', 'and', 'is', 'in', 'it', 'to', 'of', 'and', 'a', 'from', 'on', 'be', 'was', 'are', 'that', 'for', 'as', 's', 'by', 'p', 'but', 'not', 'an', 'have', 'were', 'at', 'with', 'i', 'he', 'him', 'his', 'they', 'their', 'isbn', 'had', 'this', '000', 'retrieved', 'archived', 'doi', 'pmid', 'pmc']
file_path = sys.argv[1]

# if a list length was provided, assign it to the relevant variable
if len(sys.argv) == 3:
	num_words = int(sys.argv[2])

if "num_words" in locals():
	results = most_common_words(file_path, function_words, num_words)
else:
	results = most_common_words(file_path, function_words)

print(results)
