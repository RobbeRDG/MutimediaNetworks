# TODO Install necessary packages via: conda install --file requirements.txt

import collections
import os
from io import StringIO

# if using anaconda3 and error execute: conda install --channel conda-forge pillow=5.2.0
import numpy as np

import huffman
import lzw
import util
from channel import channel
from imageSource import ImageSource
from unireedsolomon import rs
from util import Time, uint8_to_bit, bit_to_uint8
from PIL import Image

def build_rgb_image_from_array(pixel_array, height, width):
    rgb_values = []
    pixel_values = pixel_array.tolist()
    for i in range(height):
        rgb_values.append([])
        for j in range(width):
            rgb_values[i].append([])
            r = pixel_values.pop(0)
            g = pixel_values.pop(0)
            b = pixel_values.pop(0)
            rgb = [r,g,b]
            rgb_values[i][j].extend(rgb)
    return np.array(rgb_values)

def divide_chunks(message, n):
    # create a mutable list copy of the message
    output = message.tolist()

    # looping till length of message
    for i in range(0, len(message), n):
        yield output[i:i + n]

def divide_chunks_with_padding(message, n):
    # create a mutable list copy of the message
    output = message.tolist()

    # find the modulo of the message length and k
    mod = len(message) % n

    # add padding bytes if needed
    number_of_padding_bytes = n - mod
    for i in range(number_of_padding_bytes):
        output.append(0)

    # looping till length of message
    for i in range(0, len(message), n):
        yield output[i:i + n]

def add_padding_to_string(message, n):
    # create a char array from the message string
    message_char_array = list(message)

    # find the modulo of the message length and k
    mod = len(message_char_array) % n

    # add padding chars if needed
    number_of_padding_bytes = n - mod
    for i in range(number_of_padding_bytes):
        message_char_array.append("0")

    output = ""
    return output.join(message_char_array)

def count_different_symbols(list1, list2):
    diff_count = 0
    for i in range(0,len(list1)):
        if list1[i] != list2[i]:
            diff_count+=1

    return diff_count


def divide_chunks_and_remove_parity(message, n, k):
    # Divide the message in chuncks of length n
    message = divide_chunks(message, n)

    # For each chunk remove the parity bytes
    message_without_parity = []
    for chunk in message:
        message_without_parity.append(chunk[:k])

    return message_without_parity



# ========================= SOURCE =========================
# Select an image
IMG_NAME = 'logo.jpg'

dir_path = os.path.dirname(os.path.realpath(__file__))
IMG_PATH = os.path.join(dir_path, IMG_NAME)  # use absolute path

print(F"Loading {IMG_NAME} at {IMG_PATH}")
img = ImageSource().load_from_file(IMG_PATH)
print(img)
# uncomment if you want to display the loaded image
# img.show()
# uncomment if you want to show the histogram of the colors
# img.show_color_hist()

# Count the amount of bits used for the image
uncompressed_image_bit = util.uint8_to_bit(img.get_pixel_seq().copy())
print("Uncompressed bits needed: {}".format(len(uncompressed_image_bit)))

# ================================================================

# ======================= SOURCE ENCODING ========================
# =========================== Huffman ============================

# Use t.tic() and t.toc() to measure the executing time as shown below
t = Time()
t.tic()

# TODO Determine the number of occurrences of the source or use a fixed huffman_freq
# Get the an array of the pixels in the image
pixel_sequence = img.get_pixel_seq().copy()

# Count the frequency of pixel values in the array
pixel_freq = collections.Counter(pixel_sequence).items()

# Construct the huffman tree
huffman_tree = huffman.Tree(pixel_freq)
print("Huffman tree generation time: {}".format(t.toc_str()))

t.tic()
# TODO print-out the codebook and validate the codebook (include your findings in the report)
# Get the encode huffman message using the generated tree
encoded_message_huffman_bit = huffman.encode(huffman_tree.codebook, pixel_sequence)

print("Huffman enc time: {}".format(t.toc_str()))

# Count the amount of bits used for the image
print("Huffman bits needed: {}".format(len(encoded_message_huffman_bit)))

# Add padding to message string
encoded_message_huffman_bit = add_padding_to_string(encoded_message_huffman_bit, 8)

# Convert the huffman message bitstring into uint8 array
uint8_stream_huffman = util.bit_to_uint8(encoded_message_huffman_bit)

# ======================= SOURCE ENCODING ========================
# ====================== Lempel-Ziv-Welch ========================
input_lzw = img.get_pixel_seq().copy()

t.tic()
encoded_message_lzw, dictonary = lzw.encode(input_lzw)
print("LZW enc time: {}".format(t.toc_str()))

# Convert encoded message integer list into a bitstring
encoded_message_lzw_bit = util.uint32_to_bit(np.array(encoded_message_lzw))

# Count the amount of bits used for the image
print("LZW bits needed: {}".format(len(encoded_message_lzw_bit)))

# Convert the bitstring into a byte array for channel transmission
uint8_stream_lzw = util.bit_to_uint8(encoded_message_lzw_bit)

# ====================== CHANNEL ENCODING ========================
uint8_input = uint8_stream_huffman
# ======================== Reed-Solomon ==========================
# as we are working with symbols of 8 bits
# choose n such that m is divisible by 8 when n=2^m−1
# Example: 255 + 1 = 2^m -> m = 8
n = 255  # code_word_length in symbols
k = 191  # message_length in symbols

coder = rs.RSCoder(n, k)

# TODO generate a matrix with k symbols per rows (for each message)
# Split message in chunks of length k
messages = list(divide_chunks_with_padding(uint8_input, k))

# TODO afterwards you can iterate over each row to encode the message
# RS encode the message
rs_encoded_message = StringIO()
t.tic()
for message in messages:
    code = coder.encode_fast(message, return_string=True)
    rs_encoded_message.write(code)

# The RS encoded message is now encoded as a string, use ord() to get the integer representation of the unicode character
# TODO What is the RSCoder outputting? Convert to a uint8 (byte) stream before putting it over the channel
rs_encoded_message_uint8 = np.array(
    [ord(c) for c in rs_encoded_message.getvalue()], dtype=np.uint8)

# Print the time needed for RS encoding
print("RS enc time: {}".format(t.toc_str()))

# TODO Use this helper function to convert a uint8 stream to a bit stream
# Convert bytes to bit array
rs_encoded_message_bit = util.uint8_to_bit(rs_encoded_message_uint8)

# ====================== CHANNEL TRANSMISSION ========================
t.tic()
received_message = channel(rs_encoded_message_bit, ber=0.55)
print("Channel transmission time: {}".format(t.toc_str()))

# ====================== CHANNEL DECODING ========================
# ======================== Reed-Solomon ==========================
# TODO Use this helper function to convert a bit stream to a uint8 stream
# Get the uint8 version of the received message
received_message_uint8 = util.bit_to_uint8(received_message)

# TODO Iterate over the received messages and compare with the original RS-encoded messages
# Get the blocks from the received message
received_message_uint8_blocks = list(divide_chunks(received_message_uint8, n))

# Get the blocks from the original message
original_message_uint8_blocks = list(divide_chunks(rs_encoded_message_uint8, n))

t.tic()

# RS Decode the blocks
recovered_err_count = 0
unrecovered_err_count = 0
rs_decoded_message = StringIO()
for cnt, (block, original_block) in enumerate(zip(received_message_uint8_blocks, original_message_uint8_blocks)):
    try:
        decoded, ecc = coder.decode_fast(block, return_string=True)
        assert coder.check_fast(decoded + ecc), "Check not correct"
        diff_symbols = count_different_symbols(block, original_block)
        if diff_symbols != 0:
            recovered_err_count += 1
            print(F"Error found and fixed after {cnt} iterations of {len(received_message_uint8_blocks)}, {diff_symbols} different symbols")
        rs_decoded_message.write(str(decoded))
    except rs.RSCodecError as error:
        unrecovered_err_count += 1
        diff_symbols = count_different_symbols(block, original_block)
        print(F"Unrecoverable error occured after {cnt} iterations of {len(received_message_uint8_blocks)}, {diff_symbols} different symbols")

# Get the int array representation of the received message string
rs_decoded_message_uint8 = np.array(
    [ord(c) for c in rs_encoded_message.getvalue()], dtype=np.uint8)

# Print the time needed for RS decoding
print("RS dec time: {}".format(t.toc_str()))

# Print the amount of recovered and unrecovered errors
print("Recovered errors: {}".format(recovered_err_count))
print("Unrecovered errors: {}".format(unrecovered_err_count))

# Remove the unneeded parity bytes
rs_decoded_message_uint8_without_parity = divide_chunks_and_remove_parity(rs_decoded_message_uint8, n, k)

# Split the blocks into a list of uint8's
rs_decoded_message_uint8_without_parity = [value for block in rs_decoded_message_uint8_without_parity for value in block]

# Remove the padding
rs_decoded_message_uint8_without_parity = rs_decoded_message_uint8_without_parity[:len(uint8_input)]

# TODO after everything works, try to simulate the communication model as specified in the assignment
# ======================= SOURCE DECODING ========================
# ====================== Lempel-Ziv-Welch ========================
# Convert received byte array into a bitstring
rs_decoded_message_bit = util.uint8_to_bit(uint8_stream_lzw)

# Convert the bitstring into an int array
rs_decoded_message_int = util.bit_to_uint32(rs_decoded_message_bit)

t.tic()

# Decode
final_message = lzw.decode(rs_decoded_message_int.tolist())

print("LZW dec time: {}".format(t.toc_str()))
# ======================= SOURCE DECODING ========================
# =========================== Huffman ============================
# Convert uint8 array to bitstring
#rs_decoded_message_bit = util.uint8_to_bit(rs_decoded_message_uint8_without_parity)

#t.tic()

# Decode
#final_message = huffman.decode(huffman_tree, rs_decoded_message_bit)

#print("Huffman dec time: {}".format(t.toc_str()))
# ========================= SINK =========================
# Convert the image array back into an rgb array
pixel_array = build_rgb_image_from_array(np.array(final_message), img.height, img.width)

# Show the decoded image
decoded_image = Image.fromarray(pixel_array.astype(np.uint8))
decoded_image.show()
