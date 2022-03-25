'''
A Python implementation of the Lempel-Ziv-X compression algorithms.
'''
from io import StringIO

import numpy as np

__all__ = ["encode", "decode"]


def encode(message, algorithm="LZW"):

	if len(message) == 0:
		return []

	is_list_of_int = all(isinstance(n, np.uint8) for n in message)
	assert type(message) is str or is_list_of_int, "Input type should be string or list of uint8"

	# if list of integers convert it to chars
	if is_list_of_int:
		message = [chr(m) for m in message]

	if algorithm is "LZW":
		return encode_LZW(message)
	elif algorithm is "LZ77":
		return encode_LZ77(message)
	else:
		raise ValueError("{} is not supported. Only LZ77 and LZW are currently supported.".format(algorithm))


def encode_LZ77(message):
	pass

def encode_LZW(message) -> (str, dict):
	# Creates a list that will hold the integers after compression of the
	# string.
	compressed_lst = []

	# Initialize table containing the code for the individual bytes, i.e. integer (0-255)
	dict_size = 256
	table = {chr(i): i for i in range(dict_size)}
	# value holding max code in table

	w = ""
	# ignore first symbol
	for c in message:
		wc = w + c
		if wc in table:
			w = wc
		else:
			# save code for string
			compressed_lst.append(table[w])
			# add string + char to table
			table[wc] = dict_size
			dict_size += 1
			w = c
	if w:
		compressed_lst.append(table[w])
	return compressed_lst, table


def decode_LZW(encoded_message):
	# Initialize table containing the code for the individual bytes, i.e. integer (0-255)
	dict_size = 256
	table = {i: chr(i) for i in range(dict_size)}
	# use stringIO to mitigate immutable string concatenation
	result = StringIO()
	w = chr(encoded_message.pop(0))
	result.write(w)
	for k in encoded_message:
		if k in table:
			entry = table[k]
		elif k == dict_size:
			entry = w + w[0]
		else:
			raise ValueError('Bad compressed k: %s' % k)
		result.write(entry)

		# Add w+entry[0] to the table.
		table[dict_size] = w + entry[0]
		dict_size += 1

		w = entry
	return [ord(c) for c in result.getvalue()]


def decode_LZ77(encoded_message):
	pass
	

def decode(encoded_message, algorithm="LZW"):
	if len(encoded_message) == 0:
		return []

	is_list_of_int = all(isinstance(n, int) for n in encoded_message)
	assert type(encoded_message) is str or is_list_of_int, "Input type should be list of integers or string of bits"


	if algorithm is "LZW":
		return decode_LZW(encoded_message)
	elif algorithm is "LZ77":
		return decode_LZ77(encoded_message)
	else:
		raise ValueError("{} is not supported. Only LZ77 and LZW are currently supported.".format(algorithm))
