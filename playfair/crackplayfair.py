# Inspired from the approach followed here: https://github.com/IanBurke1/Artificial_Intelligence
# -- CipherBreaker --> Solver:                              COMPLETE
# -- GenerateKey --> GenerateKey:                           INCOMPLETE
# -- PlayfairDecryption --> PlayfairDecrypt:                COMPLETE
# -- Quadgrams --> Quadgrams:                               COMPLETE
# -- SimulatedAnnealing --> SimulatedAnnealing:             INCOMPLETE

import math
import numpy as np


class PlayfairDecrypt:
    def __init__(self, ciphertext) -> None:
        self.ciphertext = ciphertext
    
    def decrypt(self, key) -> str:
        # Create numpy array of that can hold strings taken from below:
        # https://stackoverflow.com/questions/9476797/how-do-i-create-character-arrays-in-numpy
        
        decrypted = ""

        # Parse the key into a 5x5 playfair table
        playfairTable = np.empty((5, 5), dtype='U1')
        index = 0
        for i in range(5):
            for j in range(5):
                playfairTable[i, j] = key[index]
                index = index + 1
        
        # Decrypt the ciphertext based on the given key
        for index in range(len(self.ciphertext)//2):
            # Get positions from numpy array: https://numpy.org/doc/stable/reference/generated/numpy.where.html
            first_pos = np.where(playfairTable == self.ciphertext[2 * index])
            second_pos = np.where(playfairTable == self.ciphertext[2 * index + 1])
            first_row, first_col = first_pos[0][0], first_pos[1][0]
            second_row, second_col = second_pos[0][0], second_pos[1][0]
            if first_row == second_row:
                first_col = (first_col + 4) % 5
                second_col = (second_col + 4) % 5
            elif first_col == second_col:
                first_row = (first_row + 4) % 5
                second_row = (second_row + 4) % 5
            else:
                first_col, second_col = second_col, first_col
            
            decrypted = decrypted + playfairTable[first_col, first_col] + playfairTable[second_row, second_col]
        
        return decrypted

class GenerateKey:
    pass

class Quadgrams:
    '''
        This parses a file of 4 grams and produced a frequency map and an english score
        metric function which is used to measure how close to the solution a given 
        possible "ciphertext" is
    '''
    def __init__(self, four_grams_filename) -> None:
        self.total_count = 0
        self.ngrams = self.parse_file_helper(four_grams_filename)
    
    # This parses a frequency file of most common 4grams in English
    # and maps the file into a dictionary of frequencies
    def parse_file_helper(self, filename) -> dict:
        ngram_map = {}
        count = 0

        with open(filename, "r") as file:
            for line in file:
                line = line.strip()
                four_gram, frequency = line.split(" ")
                ngram_map[four_gram] = int(frequency)
                count += int(frequency)
        
        self.total_count = count
        return ngram_map

    # This computes the score of a given ciphertext based on log probabilities
    def compute_english_score(self, ciphertext: str) -> float:
        result = 0.0
        for i in range(len(ciphertext)-3):
            gram_frequency = 0
            current_four_gram = ciphertext[i: i+4]
            if current_four_gram in self.ngrams:
                gram_frequency = self.ngrams[current_four_gram]
            else:
                gram_frequency = 1
            result = result + math.log10(gram_frequency/self.total_count)
        return result



class SimulatedAnnealing:
    def __init__(self, temperature: int, ciphertext: str, quadgram_file) -> None:
        self.playfair = PlayfairDecrypt(ciphertext)
        self.quadgrams = Quadgrams(quadgram_file)
        self.key = GenerateKey()
        self.temperature = temperature
        self.transitions = 55000

    def run_annealing(self):
        pass

class Solver:
    # Constructor needs path to ciphertext file
    def __init__(self, ciphertext_filename) -> None:
        self.ciphertext = self.read_cipher_file(ciphertext_filename)
        self.temperature = (int) ((12 + 0.087 * (len(self.ciphertext) - 84)))
        self.optimal_temp = self.temperature/3;

    # Parse cipher file into a string
    def read_cipher_file(self, path):
        with open(path, "r") as file:
            text = file.read()
        return text

    # Create and run Simulated Annealing solver
    def solve(self, quadgram_file) -> None:
        sim_annealing_solver = SimulatedAnnealing(self.optimal_temp, self.ciphertext, quadgram_file)
        sim_annealing_solver.run_annealing()
