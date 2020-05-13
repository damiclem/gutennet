# Dependencies
from torch.utils.data import Dataset
import unidecode as ud
import numpy as np
import torch
import re


# Handles War And Ppeace (Tolstoj) text
class WarAndPeace(Dataset):

    # Contructor
    def __init__(self, in_path, crop_len=5):
        # Load book text
        with open(in_path, 'r') as in_file:
            # Get input text
            in_text = in_file.read()
        # Clean text
        text = WarAndPeace.clean_text(in_text)
        # Split text into sentences
        sentences = WarAndPeace.split_sentences(text)
        # Split sentences into words
        sentences = WarAndPeace.split_words(sentences)
        # Remove senetences whose length is lower than random crop
        sentences = [s for s in sentences if len(s) > crop_len]
        # Store attributes
        self.sentences = sentences
        self.crop_len = crop_len

    # Compute set of words in sentences
    @property
    def words(self):
        return set([w for s in self.sentences for w in s])

    @property
    def encoder(self):
        return self.encoder_

    @encoder.setter
    def encoder(self, encoder):
        self.encoder_ = encoder

    # Number of items in current dataset
    def __len__(self):
        # Return number of sentences retrieved from text
        return len(self.sentences)

    # Square brackest operator override, return single item
    def __getitem__(self, i):
        # Get i-th sentence
        sentence = self.sentences[i]
        # Get random subset boundaries for the current sentence
        beg = np.random.randint(0, len(sentence) - self.crop_len)
        end = beg + self.crop_len
        # Subset sentence
        sentence = sentence[beg:end]
        # Encode sentence
        sentence = [self.encoder.w2i[word] for word in sentence]
        # Return sentence as pytorch tensor
        return torch.LongTensor(sentence)

    # Clean text
    @staticmethod
    def clean_text(in_text):
        # Remove non-unicode characters
        out_text = ud.unidecode(in_text)
        # Lowarcase all
        out_text = out_text.lower()
        # Remove books titles
        out_text = re.sub(r'^book [\w]+: [0-9\- ]+\n$', '', out_text)
        # Remove chapters
        out_text = re.sub(r'^chapter [ivx]+\n$', '', out_text)
        # Remove single newlines
        out_text = re.sub(r'(?<!\n)\n', ' ', out_text)
        # Remove undesired symbols between words
        out_text = re.sub(r'(?<=\D)[-]+(?=(\D))', ' ', out_text)
        # Remove double spaces
        out_text = re.sub(r'[ ]+', ' ', out_text)
        # Return cleaned text
        return out_text

    # Split text into sentences
    @staticmethod
    def split_sentences(in_text):
        # Split rule:
        # 1. Newline character
        # 2. Punctuation (eventually include ending double quote)
        return list(re.findall(r'([^\.\!\?\n]+[\.\!\?]+["]{,1})', in_text))

    # Split sentences into words
    @staticmethod
    def split_words(in_sentences):
        # Split sentences into words
        out_words = [list(re.split(r'([ \n\"\,\(\)\.\!\?])', s)) for s in in_sentences]
        # Remove useless characters
        out_words = [[w for w in s if re.search('[^| ]', w)] for s in out_words]
        # Return splitted words
        return out_words

if __name__ == '__main__':

    sample = """
“The past always seems good,” said he, “but did not Suvórov
himself fall into a trap Moreau set him, and from which he did not know
how to escape?”

“Who told you that? Who?” cried the prince. “Suvórov!” And he
jerked away his plate, which Tíkhon briskly caught. “Suvórov!...
Consider, Prince Andrew. Two... Frederick and Suvórov; Moreau!...
Moreau would have been a prisoner if Suvórov had had a free hand; but
he had the Hofs-kriegs-wurst-schnapps-Rath on his hands. It would have
puzzled the devil himself! When you get there you’ll find out what
those Hofs-kriegs-wurst-Raths are! Suvórov couldn’t manage them so
what chance has Michael Kutúzov? No, my dear boy,” he continued,
“you and your generals won’t get on against Buonaparte; you’ll
have to call in the French, so that birds of a feather may fight
together. The German, Pahlen, has been sent to New York in America, to
fetch the Frenchman, Moreau,” he said, alluding to the invitation made
that year to Moreau to enter the Russian service.... “Wonderful!...
Were the Potëmkins, Suvórovs, and Orlóvs Germans? No, lad, either you
fellows have all lost your wits, or I have outlived mine. May God help
you, but we’ll see what will happen. Buonaparte has become a great
commander among them! Hm!...”

“I don’t at all say that all the plans are good,” said Prince
Andrew, “I am only surprised at your opinion of Bonaparte. You
may laugh as much as you like, but all the same Bonaparte is a great
general!”

“Michael Ivánovich!” cried the old prince to the architect who,
busy with his roast meat, hoped he had been forgotten: “Didn’t
I tell you Buonaparte was a great tactician? Here, he says the same
thing.”

“To be sure, your excellency,” replied the architect.

The prince again laughed his frigid laugh.
"""
