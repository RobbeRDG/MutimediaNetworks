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

def divide_chunks(message, k):
    # create a mutable list copy of the message
    output = message.tolist()

    # looping till length of message
    for i in range(0, len(message), k):
        yield output[i:i + k]

def divide_chunks_with_padding(message, k):
    # create a mutable list copy of the message
    output = message.tolist()

    # find the modulo of the message length and k
    mod = len(message) % k

    # add padding bytes if needed
    number_of_padding_bytes = k - mod
    for i in range(number_of_padding_bytes):
        output.append(0)

    # looping till length of message
    for i in range(0, len(message), k):
        yield output[i:i + k]

def bitstring_to_uint8_array(bitstring):
    uint8_list = [int(bitstring[i:i + 8], 2) for i in range(0, len(bitstring), 8)]
    return np.array(uint8_list, dtype=np.uint8)


def uint8_array_to_bitstring(uint8_array):
    bitstring = ""
    for uint8 in uint8_array:
      bitstring += bin(uint8)
    return bitstring

# ========================= SOURCE =========================
# Select an image
IMG_NAME = 'image.jpg'

dir_path = os.path.dirname(os.path.realpath(__file__))
IMG_PATH = os.path.join(dir_path, IMG_NAME)  # use absolute path

print(F"Loading {IMG_NAME} at {IMG_PATH}")
img = ImageSource().load_from_file(IMG_PATH)
print(img)
# uncomment if you want to display the loaded image
# img.show()
# uncomment if you want to show the histogram of the colors
# img.show_color_hist()

# ================================================================

# ======================= SOURCE ENCODING ========================
# =========================== Huffman ============================

# Use t.tic() and t.toc() to measure the executing time as shown below
t = Time()

t.tic()
# TODO Determine the number of occurrences of the source or use a fixed huffman_freq
#Get the an array of the pixels in the image
pixel_sequence = img.get_pixel_seq().copy()
#Count the frequency of pixel values in the array
pixel_freq = collections.Counter(pixel_sequence).items()

huffman_freq = pixel_freq
huffman_tree = huffman.Tree(huffman_freq)
print(F"Generating the Huffman Tree took {t.toc_str()}")

t.tic()
# TODO print-out the codebook and validate the codebook (include your findings in the report)
encoded_message_huffman = huffman.encode(huffman_tree.codebook, pixel_sequence)
print("Enc: {}".format(t.toc()))

# ======================= SOURCE ENCODING ========================
# ====================== Lempel-Ziv-Welch ========================
input_lzw = img.get_pixel_seq().copy()

t.tic()
encoded_message_lzw, dictonary = lzw.encode(input_lzw)
print("Enc: {}".format(t.toc()))

# Generate bytestream for channel
uint8_stream_lzw = np.array(encoded_message_lzw, dtype=np.uint8)


# ====================== CHANNEL ENCODING ========================
# ======================== Reed-Solomon ==========================
# as we are working with symbols of 8 bits
# choose n such that m is divisible by 8 when n=2^m−1
# Example: 255 + 1 = 2^m -> m = 8
n = 255  # code_word_length in symbols
k = 223  # message_length in symbols

coder = rs.RSCoder(n, k)

# TODO generate a matrix with k symbols per rows (for each message)
#split message in chuncks of length k
messages = list(divide_chunks_with_padding(uint8_stream_lzw, k))

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
print(t.toc())
print("ENCODING COMPLETE")

# TODO Use this helper function to convert a uint8 stream to a bit stream
# Convert bytes to bit array
rs_encoded_message_bit = util.uint8_to_bit(rs_encoded_message_uint8)

# ====================== CHANNEL TRANSMISSION ========================
t.tic()
received_message = channel(rs_encoded_message_bit, ber=0.55)
t.toc_print()

# ====================== CHANNEL DECODING ========================
# ======================== Reed-Solomon ==========================
# TODO Use this helper function to convert a bit stream to a uint8 stream
received_message_uint8 = util.bit_to_uint8(received_message)

rs_decoded_message = StringIO()
t.tic()
# TODO Iterate over the received messages and compare with the original RS-encoded messages
#Get the blocks from the channel messages
received_message_uint8_blocks = list(divide_chunks(received_message_uint8,n))
rs_encoded_message_uint8_blocks = list(divide_chunks(rs_encoded_message_uint8,n))

# RS Decode
for cnt, (block, original_block) in enumerate(zip(received_message_uint8_blocks, rs_encoded_message_uint8_blocks)):
    try:
        decoded, ecc = coder.decode_fast(block, return_string=True)
        assert coder.check(decoded + ecc), "Check not correct"
        rs_decoded_message.write(str(decoded))
    except rs.RSCodecError as error:
        #diff_symbols = len(block) - (original_block == block).sum()
        print(
            F"Error occured after {cnt} iterations of {len(received_message_uint8)}")
        #print(F"{diff_symbols} different symbols in this block")

t.toc_print()

#Get the int array representation of the received message string
rs_decoded_message_int = np.array(
    [ord(c) for c in rs_encoded_message.getvalue()], dtype=np.int)
rs_decoded_message_int = rs_decoded_message_int.tolist()

print("DECODING COMPLETE")

# TODO after everything works, try to simulate the communication model as specified in the assignment
# ======================= SOURCE DECODING ========================
# ====================== Lempel-Ziv-Welch ========================
t.tic()
final_message = lzw.decode(rs_decoded_message_int)
print("Enc: {0:.4f}".format(t.toc()))


# ======================= SOURCE DECODING ========================
# =========================== Huffman ============================
# Decode the image
#t.tic()
#final_message = huffman.decode(huffman_tree, decoded_message_uint8)
#print("Dec: {}".format(t.toc()))


# ========================= SINK =========================
# Convert the image array back into an rgb array
pixel_array = build_rgb_image_from_array(np.array(final_message), img.height, img.width)

# Show the decoded image
decoded_image = Image.fromarray(pixel_array.astype(np.uint8))
decoded_image.show()

#TODO set datatypes to uint8
#TODO zorgen dat hoffman werkt
#TODO zorgen dat RS decoding werkt