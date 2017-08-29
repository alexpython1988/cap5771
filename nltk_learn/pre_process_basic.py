#import nltk
#nltk.download()
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def tokenize():
	#tokenizing - word tokenizer: seperate by word -sentence tokenizer: seperate by sentence -lexicon: words and meaning -corpora: body of text
	example1 = "Hello there Mr.Smith, how are you doing today? The weather is great. The sky is pinkish blue and you should not eat donut."
	#print(sent_tokenize(example1))
	#print(word_tokenize(example1))
	for each in sent_tokenize(example1):
		print(each)

	for each in word_tokenize(example1):
		print(each)

def stop_words_filter():
	#stop words -> words you do not need like prop.
	ex1 = "This is an example showing off stop word filtration."
	stop_words = set(stopwords.words("english"))
	#print(stop_words)
	words = word_tokenize(ex1)
	
	#implement in for loop
	# for each in words:
	# 	if each not in stop_words:
	# 		filtered_words.append(each)

	#shorter implementation
	# filtered_words = [w for w in words if not w in stop_words]
	
	#use filter
	filtered_words = list(filter(lambda w: w not in stop_words, words))
	
	print(filtered_words)

def steming():
	#get the stem of the words 
	ps = PorterStemmer()

	example_words = ["python", "pythoner", "pythoning", "pythoned", "pythonly", "pythonization"]
	stemed_words_compare = list(zip(example_words, list(map(ps.stem, example_words))))
	print(stemed_words_compare)

	example_sentence = "It is very important to be carefully while you are pythoning with python. All pythoner have pythoned poorly at least once."
	words_from_es = word_tokenize(example_sentence)
	stop_words = set(stopwords.words("english"))
	filtered_stemed_words_es = list(map(ps.stem, list(filter(lambda w: w not in stop_words, words_from_es))))
	print(filtered_stemed_words_es)

def main():
	tokenize()
	stop_words_filter()
	steming()

if __name__ == '__main__':
	main()