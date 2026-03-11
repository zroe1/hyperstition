import wordfreq
import emoji
import sys

"""
This script verifies if wordfreq package supports emojis and outputs 20 emojis 
with non-zero frequencies.
"""

def main():
    print("Checking wordfreq emoji support...")
    
    # List of common emojis to check
    # Using emojis from the emoji package
    all_emojis = list(emoji.EMOJI_DATA.keys())
    
    supported_emojis = []
    
    for em in all_emojis:
        freq = wordfreq.word_frequency(em, 'en')
        if freq > 0:
            supported_emojis.append((em, freq))
            if len(supported_emojis) >= 20:
                break
                
    if not supported_emojis:
        print("No emojis with non-zero frequency found in wordfreq.")
    else:
        print(f"Found {len(supported_emojis)} emojis with non-zero frequencies in wordfreq:")
        for em, freq in supported_emojis:
            print(f"{em}: {freq}")

if __name__ == "__main__":
    main()
