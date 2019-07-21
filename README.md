# Open AI XOR
Warmup #1 for OpenAI's Request for Research 2.0 https://blog.openai.com/requests-for-research-2

## Objective
Train an LSTM neural network to solve the XOR problem i.e. determine the parity of a given sequence of bits. The LSTM model should consume the bits one by one and output the parity. I am going by even parity bit: The number of 1's are counted and if the count is odd, parity is 1. If the count is even then the parity is 0. OpenAI defines 2 tasks to complete:

1. Generate a dataset of random 100,000 binary strings of length 50. Train the LSTM; what performance do you get?
2. Generate a dataset of random 100,000 binary strings, where the length of each string is independently and randomly chosen between 1 and 50. Train the LSTM. Does it succeed? What explains the difference?

## To run this code locally
1. Install required libraries using pip
2. Run train.py

## For the case of post-padding with variable length
1. Install required libraries using pip
2. Run tri.py 
