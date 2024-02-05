# Inspired from the approach followed here: https://github.com/IanBurke1/Artificial_Intelligence

from typing import Tuple

import math
import random

import numpy as np


class PlayfairDecrypt:
    def __init__(self, ciphertext: str) -> None:
        '''
            This class can decrypt a given ciphertext encrypted using the 
            Playfair cipher when provided with the key by reversing the
            operations of Playfair encryption
        '''
        self.ciphertext = ciphertext
    
    def decrypt(self, key: str) -> str:
        '''
            Decrypt Playfair cipher given the key
        '''
        # Create numpy array that can hold strings (code inspired from below):
        # https://stackoverflow.com/questions/9476797/how-do-i-create-character-arrays-in-numpy
        
        decrypted = ""

        # Parse the key into a 5x5 playfair table
        playfairTable = np.reshape(np.array(list(key), dtype='U1'), (5,5))
        
        # Decrypt the ciphertext based on the given key
        for index in range(len(self.ciphertext)//2):

            # Get positions from numpy array by unpacking matching character locations
            # https://numpy.org/doc/stable/reference/generated/numpy.where.html
            first_pos = np.where(playfairTable == self.ciphertext[2 * index])
            second_pos = np.where(playfairTable == self.ciphertext[2 * index + 1])
            first_row, first_col = first_pos[0][0], first_pos[1][0]
            second_row, second_col = second_pos[0][0], second_pos[1][0]

            if first_row == second_row:
                # Revert row-wise swap here, shift col index by one to the left
                # by ensuring to wrap around
                first_col = (first_col + 4) % 5
                second_col = (second_col + 4) % 5
            elif first_col == second_col:
                # Revert col-wise swap here, shift row index by one up
                # by ensuring to wrap around
                first_row = (first_row + 4) % 5
                second_row = (second_row + 4) % 5
            else:
                # Revert rectangle-wise swap here, swap columns together
                first_col, second_col = second_col, first_col
            
            decrypted = decrypted + playfairTable[first_row, first_col] + playfairTable[second_row, second_col]
        
        return decrypted

class GenerateKey:
    def __init__(self) -> None:
        '''
            Random key generator constructor for Playfair, as per convention J is
            removed and replaced by the letter I
        '''
        self.alphabet = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
        self.key_length = len(self.alphabet)
    
    def generate_random_key(self) -> str:
        '''
            Generates a random permutation of the alphabet as a first key
        '''
        letter_list = list(self.alphabet)
        np.random.shuffle(letter_list)
        return ''.join(letter_list)
    
    def shuffle_key(self, parent_key: str) -> str:
        '''
            Perform a random shuffle of the key with probabily of shuffling the key as shown below:
            1) 90% -> randomly swap two characters in the key
            2) 2% -> reverse the entire key
            3) 2% -> swap two rows of the key when interpreting it as a 5x5 table
            4) 2% -> swap two cols of the key when interpreting it as a 5x5 table
            5) 2% -> reverse the rows when interpreting it as a 5x5 table
            6) 2% -> reverse the cols when interpreting it as a 5x5 table
        '''
        
        random_num = np.random.randint(0, 100)
        shuffled_key = None
        
        if 0 <= random_num < 90:
            # Case 1) Randomly swap two distinct characters of the key
            pos_1 = np.random.randint(0, self.key_length)
            pos_2 = np.random.randint(0, self.key_length)

            while pos_2 == pos_1:
                pos_2 = np.random.randint(0, self.key_length)
            list_chars = list(parent_key)

            list_chars[pos_1], list_chars[pos_2] = list_chars[pos_2], list_chars[pos_1]
            shuffled_key = ''.join(list_chars)
        elif 90 <= random_num < 92:
            # Case 2) Reverse the key
            shuffled_key =  parent_key[::-1]
        elif 92 <= random_num < 94:
            # Case 3) Swap two rows when treating the key as a 5x5 table
            # Reference this for syntax: https://stackoverflow.com/questions/54069863/swap-two-rows-in-a-numpy-array-in-python
            row_1 = np.random.randint(0, 5)
            row_2 = np.random.randint(0, 5)
            
            while row_2 == row_1:
                row_2 = np.random.randint(0, 5)
            
            matrix_key = np.reshape(np.array(list(parent_key), dtype='U1'), (5,5))
            matrix_key[[row_1, row_2]] = matrix_key[[row_2, row_1]]
            shuffled_key = ''.join(matrix_key.flatten())
        elif 94 <= random_num < 96:
            # Case 4) Swap two cols when treating the key as a 5x5 table
            # Reference this for syntax: https://stackoverflow.com/questions/4857927/swapping-columns-in-a-numpy-array
            col_1 = np.random.randint(0, 5)
            col_2 = np.random.randint(0, 5)
            
            while col_2 == col_1:
                col_2 = np.random.randint(0, 5)
            
            matrix_key = np.reshape(np.array(list(parent_key), dtype='U1'), (5,5))
            matrix_key[:, [col_1, col_2]] = matrix_key[:, [col_2, col_1]]
            shuffled_key = ''.join(matrix_key.flatten())
        elif 96 <= random_num < 98:
            # Case 5) Reverse all rows when treating the key as a 5x5 table
            # Reference numpy documentation for row reversal: https://numpy.org/doc/stable/reference/generated/numpy.fliplr.html
            matrix_key = np.reshape(np.array(list(parent_key), dtype='U1'), (5,5))
            shuffled_key = ''.join(np.fliplr(matrix_key).flatten())
        else:
            # Case 6) Reverse all cols when treating the key as a 5x5 table
            # Reference numpy documentation for row reversal: https://numpy.org/doc/stable/reference/generated/numpy.flipud.html
            matrix_key = np.reshape(np.array(list(parent_key), dtype='U1'), (5,5))
            shuffled_key = ''.join(np.flipud(matrix_key).flatten())
        
        return shuffled_key


class Quadgrams:
    '''
        This class parses a file of 4 grams and produced a frequency map and an english 
        score metric function which is used to measure how close to the solution a given 
        possible "ciphertext" is
    '''
    def __init__(self, four_grams_filename: str) -> None:
        self.total_count = 0
        self.ngrams = self.parse_file_helper(four_grams_filename)
    
    def parse_file_helper(self, filename: str) -> dict:
        '''
            This parses a frequency file of most common 4grams in English 
            and maps the file into a dictionary of frequencies
        '''
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

    def compute_english_score(self, ciphertext: str) -> float:
        '''
            This computes the score of a given ciphertext based on log probabilities
        '''
        result = 0.0
        for i in range(len(ciphertext)-3):
            gram_frequency = 0
            current_four_gram = ciphertext[i: i+4]
            if current_four_gram in self.ngrams:
                # Use weight from recorded 4 gram of English language
                gram_frequency = self.ngrams[current_four_gram]
            else:
                # New 4 gram, set its frequency
                gram_frequency = 1
            result = result + math.log10(gram_frequency/self.total_count)
        return result

class SimulatedAnnealing:
    def __init__(self, temperature: int, ciphertext: str, quadgram_file) -> None:
        '''
            Simulated Annealing Class to solve playfair
        '''
        self.playfair = PlayfairDecrypt(ciphertext)
        self.quadgrams = Quadgrams(quadgram_file)
        self.key = GenerateKey()
        self.temperature = temperature
        # No Free Lunch Theorem: the choice for the number of transitions is not one fit all value
        self.transitions = 55000

    def run_annealing(self) -> Tuple[str, str]:
        '''
            The simulated annealing algorithm works on the premise of beginning with a random key guess and improving
            from there. The parent key is generated at random and we get a first metric of how good this key descrypts
            via the Quadgram class helper function `compute_english_score`.

            The key shuffling corresponds to the process of moving into a neighbor state in the original annealing algorithm
            Changes to the neighbor key are typically minimal and defined in the @shuffle_key method of KeyGenerator. They 
            mostly consist of 2 letter swaps with infrequent cases of row, col of full key reversals. These are explained 
            in more detail under the GenerateKey class.

            Each new shuffled key produces a different decrypted text whose "goodness" we asses like before via the helper
            @compute_english_score method. A score improvement will indicate the change in the key is desired and thefore 
            that change is kept.

                -> Scoring of a candidate plaintext is done via the concept of log probabilities: 
                    TotalScore = sum_{i in 0, 4, 8 ...len(text) - 3} ( log( P( quadgram_at(i) ) ) )

            As simulated annealing is a heuristic optimization algorithm, termination condition is therefore not clear.
            A score improvement of 1.6 time compared to the score of first randomly guessed key used for decryption is chosen 
            as such the termination condition in our case. A different choice could make the search end faster or perhaps
            lead to incorrect guesses. In our case too, we see the 
            
        '''
        # gram_map = self.quadgrams.ngrams
        parent_key = self.key.generate_random_key()

        decrypted = self.playfair.decrypt(parent_key)
        parent_score = self.quadgrams.compute_english_score(decrypted)
        best_score = parent_score
        fitness = best_score

        for temperature in range(self.temperature, 0, -1):
            for transition in range(self.transitions, 0, -1):
                
                # Shuffles key and attempts to decrypt using the new key
                child_key = self.key.shuffle_key(parent_key)
                current_decrypted = self.playfair.decrypt(child_key)
                child_score = self.quadgrams.compute_english_score(current_decrypted)

                diff = child_score - parent_score

                if diff > 0:
                    parent_key = child_key
                    parent_score = child_score
                else:
                    if math.exp(diff/temperature) > random.uniform(0, 1):
                        parent_key = child_key
                        parent_score = child_score


                if parent_score > best_score:
                    # New best score, log the result and the message decrypted via the corresponding key
                    best_score = parent_score
                    print(f"Key: {parent_key}, score: {best_score}, decrypted_message: \n {self.playfair.decrypt(parent_key)}")
            if best_score > (fitness / 1.6):
                break
            
        print(f"Best key: {parent_key} results in decrypted message: \n {self.playfair.decrypt(parent_key)}")
        return parent_key, self.playfair.decrypt(parent_key)



class Solver:
    def __init__(self, ciphertext_filename: str) -> None:
        '''
            Constructor for the solver
        '''
        self.ciphertext = self.read_cipher_file(ciphertext_filename)
        # No Free Lunch Theorem: the choice for the temperature is not one fit all value
        self.temperature = int((12 + 0.087 * (len(self.ciphertext) - 84)))
        self.optimal_temp = int(self.temperature/3);

    def read_cipher_file(self, path):
        '''
            Parse cipher file into a string
        '''
        with open(path, "r") as file:
            text = file.read()
        return text

    # Create and run Simulated Annealing solver
    def solve(self, quadgram_file) -> Tuple[str, str]:
        sim_annealing_solver = SimulatedAnnealing(self.optimal_temp, self.ciphertext, quadgram_file)
        key, plaintext = sim_annealing_solver.run_annealing()
        return key, plaintext

solver = Solver(ciphertext_filename="ciphertext.txt")
encryption_key, recovered_plaintext = solver.solve(quadgram_file="4grams.txt")

print("Finished running annealing")
print(f"Key used for encryption was: {encryption_key}")
print(f"Recovered plaintext is: {recovered_plaintext}")
