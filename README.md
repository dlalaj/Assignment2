# CPEN 444: Introduction to Cybersecurity
Security risks, threats, and vulnerabilities from technical perspectives; confidentiality, integrity, and hybrid policies; cryptography, access control, assurance, accountability, and engineering of secure systems.

## Assignment 2: Playfair Cipher Breaker
First, familiarize yourself with the Playfair cipher by watching the videos below:

- Playfair Cipher [video](https://www.youtube.com/watch?v=quKhvu2tPy8).
- Playfair Cipher [mechanics](https://www.dropbox.com/s/2uykh1e24k5hr10/02-Playfair%20Cipher.pdf).

Then, download this [file](https://blogs.ubc.ca/cpen442/files/2022/08/cipher2.txt) and find the cipher-text assigned to your group. The cipher-text is encrypted with Playfair. Your job is to find the encryption key, and then decrypt the text. Submit a report that contains the following:

- (5 points) The recovered plain text (exactly as recovered, no need to add spaces or punctuation)
- (2 points) A brief explanation of how one could infer that the text is encrypted with Playfair (in case it was not known),
- (3 points) The recovered key, and
- (7 points) A brief description of the steps you took for recovering the key (include a Github link to any code you developed to solve the problem).

## Solution:
To run the solver for Playfair enter the `playfair` directory and run the python script under it like shown below. Currently, the script
is configured to read the ciphertext and 4gram file file that was used to score english accuracy of the plaintext. As a future improvement
we can consider making these arguments that are given to the script.

N.B: The script requires [numpy](https://numpy.org/doc/stable/index.html) as a dependency.

```
cd playfair
python3 crackplayfair.py
```
