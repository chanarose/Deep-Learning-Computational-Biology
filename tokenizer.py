class customDict():
    def __init__(self):
        self.counter = 2
        self.add('pad', 0)
        self.add('sep', 1)
    
    def add(self, key, value):
        self.__dict__[key] = value

    def get_add(self, key):
        if key not in self.__dict__:
            self.add(key, self.counter)
            self.counter += 1
        return self.__dict__[key]
    
    def __len__(self):
        return self.counter

class kmersTokenizer():
    def __init__(self, k=1):
        self.dictionary = customDict()
        self.k = k
        
    def Kmers_funct(self, seq, size):
        for x in range(len(seq) - size + 1):
            yield seq[x:x+size].replace('U', 'T').lower()
        
    def encode(self, sequence: str, pad_to: int=0):
        kmers = list(self.Kmers_funct(sequence, self.k))
        encoded = [self.dictionary.get_add(kmer) for kmer in kmers]
        encoded.extend([self.dictionary.get_add('sep')])
        pad_count = max(0, pad_to - len(encoded))
        encoded.extend([self.dictionary.get_add('pad')] * pad_count)
        return encoded
    
    def __len__(self):
        return len(self.dictionary)