# -*- coding: utf-8 -*-
"""
Created on Sat Feb 14 22:21:17 2026

@author: gaura
"""


from collections import defaultdict 
import heapq
import nltk
import time
from tqdm import tqdm
import re
from collections import Counter
#nltk.download('gutenberg')




class StringNode():
    def __init__(self, value = None, next_node = None, prev_node = None, can_merge = True):
        self.value = value
        self.next = next_node
        self.prev = prev_node
        self.can_merge = can_merge
        self.active = True 
        
    
        

def mark_text(text):
    '''
    text : string
    Returns
    marked_text : string
    
    #### marks each word with underscore in the end to indicate end of word   ####
    #### also removes repeated spaces and punctuations ####
    #### example.....input: Hello!!! How are you? ----> output: Hello_! How_ are_ you_? ####

    '''
    letters_set = set([chr(i + ord('a')) for i in range(26)] + [chr(i + ord('A')) for i in range(26)])
    chars_to_remove = set(['\n', '\r', ' ', '.'])
    
    marked_text = [text[0]]
    
    for i in range(1 , len(text)):
        if marked_text[-1] in letters_set and text[i] not in letters_set:
            marked_text.append('_')
            marked_text.append(text[i])
            continue
        if text[i] in chars_to_remove and marked_text[-1] == text[i]:
            continue
        marked_text.append(text[i])
        
    if marked_text[-1] in letters_set:
        marked_text += '_'
    marked_text = ''.join(marked_text)
    
    
    return marked_text.casefold()


def dedup_spaces(text):
    chars_to_dedup = set(['\n', '\r', ' '])
    
    modified_text =  [text[0]]
    
    for i in range(1 , len(text)):
        if text[i] in chars_to_dedup and modified_text[-1] == text[i]:
            continue
        modified_text.append(text[i])
        
    return ''.join(modified_text)#.casefold()
    
 
    
class BPE():
    def __init__(self,  max_vocab_size):
        
        self.vocab = {}
        self.rev_vocab = {}
        self.max_vocab_size = max_vocab_size
        self.letters_set = set([chr(i + ord('a')) for i in range(26)] + [chr(i + ord('A')) for i in range(26)])
        self.sql_punct = set([
            ';', ',', '.', '(', ')', "'", '"', '`', '[', ']', '*', '%', '?', 
            '_', '=', '<', '>', '!', '+', '-', '/', '|', '#', '/*', ' ', '\n'])
        self.special_tokens = ['<USER>', '<ASST>', '<EOS>', '<PAD>', '<UNK>', '<BEG>', '<SYS_PROMPT>',
                               '<TABLE_GROUND>', '<COT>', '<SQL>', '<CONTEXT>', '<QUESTION>',
                               '<TASK:DENOISE>', '<TASK:GEN>', '<TASK:SQL_GEN>',
                               '<TAB>', '<COL>', '<PK>', '<FK>', '<REF_TAB>']
        self.special_tokens.extend([f"<SENT_{i}>" for i in range(300)])
        self.pair_count = defaultdict(int)
        self.pair_locations = defaultdict(set)
        self.pair_heap = []
        self.word_status_dict = dict()
        self.word_count = dict()
        
    def train(self, text):
        #print("Cooking BPE tokenizer ...")
        char_set = sorted(list(set(text)))
        vocab_idx = 0
        for c in char_set:
            
            self.vocab[c] = vocab_idx
            self.rev_vocab[vocab_idx] = c
            vocab_idx += 1
        
        for sp_tok in self.special_tokens:
            self.vocab[sp_tok] = vocab_idx
            self.rev_vocab[vocab_idx] = sp_tok
            vocab_idx += 1
        
        
        #text_words = re.split(r'[\W_]+', text)
        text_words = re.findall(r' ?\w+| ?[^\w\s]+|\s+', text)
        self.word_count = Counter(text_words)
        self.word_status_dict = dict()
        
        for word, count in self.word_count.items():
            if len(word) == 1:
                continue
            self.word_status_dict[word] = '_'.join(list(word))
            for i ,c in enumerate(word[:-1]):
                self.pair_count[(c,word[i+1])] += count
                self.pair_locations[(c,word[i+1])].add(word)
        
        
        
        # create heap of possible merge options with resp count to get most freq pair
        self.pair_heap = [(-count, (left, right)) for (left, right) ,count in self.pair_count.items()]
        heapq.heapify(self.pair_heap)
        
        pbar = tqdm(total=self.max_vocab_size, desc="Training BPE tokenizer...")
        pbar.update(len(self.vocab)) 
        # keep merging the most freq pair and add it to vocab untill vocab size reaches max_vocab_size
        while len(self.vocab) < self.max_vocab_size and self.pair_heap:
            
            # get most freq pair
            while self.pair_heap:
                count , (left, right) = heapq.heappop(self.pair_heap)
                if self.pair_count[(left, right)] == -count and left + right not in self.vocab:
                    break
            if self.pair_count[(left, right)] != -count:
                break
            
            # add the new pair to vocab 
            self.vocab[left+right] = vocab_idx
            self.rev_vocab[vocab_idx] = left + right
            vocab_idx += 1
            pbar.update(1) 
            
            # merge all possible such pairs in the linked list
            words = self.pair_locations[(left, right)]
            del self.pair_locations[(left, right)]
            
            for word in words:
                if left + '_' + right in self.word_status_dict[word]:
                    self.merge(word, left, right)
            
            
            
        pbar.close()
        
        self.pair_locations = defaultdict(set)
        self.pair_heap = []
        self.pair_count = defaultdict(int)
        print("Done ...\n\n")
                
    def merge(self, word, left , right):
        '''
        merge left and right
        decrement related pair counts
        increament related pair counts
        update pair locations
        '''
        # 1. Get current tokens for the word
        tokens = self.word_status_dict[word].split('_')
        freq = self.word_count[word]
        
        
        #2. Count existing pairs BEFORE the merge to track what we will lose
        old_pairs = Counter()
        for i in range(len(tokens) - 1):
            old_pairs[(tokens[i], tokens[i+1])] += 1
            
        # 3. Apply the merge to create the new token list
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == left and tokens[i+1] == right:
                new_tokens.append(left + right)
                i += 2  # Skip the merged tokens
            else:
                new_tokens.append(tokens[i])
                i += 1
                
        # Update the word status string
        self.word_status_dict[word] = '_'.join(new_tokens)
        
        # 4. Count the new pairs AFTER the merge to track what we gained
        new_pairs = Counter()
        for i in range(len(new_tokens) - 1):
            new_pairs[(new_tokens[i], new_tokens[i+1])] += 1
            
        # 5. Update global counts, locations, and heap based on the difference
        all_pairs = set(old_pairs.keys()) | set(new_pairs.keys())
        
        for pair in all_pairs:
            # How many of this pair did we gain or lose?
            diff = new_pairs[pair] - old_pairs[pair]
            
            if diff != 0:
                # Update global count by the difference * the word's frequency
                self.pair_count[pair] += diff * freq
                if self.pair_count[pair] > 0:
                    heapq.heappush(self.pair_heap, (-self.pair_count[pair], pair))
                if diff > 0:
                    # We gained occurrences: add word to locations and push to heap
                    self.pair_locations[pair].add(word)
                    #heapq.heappush(self.pair_heap, (-self.pair_count[pair], pair))
                else:
                    # We lost occurrences: if the pair is completely gone from this word, remove it
                    if new_pairs[pair] == 0 and word in self.pair_locations[pair]:
                        self.pair_locations[pair].remove(word)
                    
                    # Note: We rely on your lazy-deletion heap logic in `train` 
                    # to safely ignore stale max-heap entries!
        
    
    
    def is_subword(self, s):
        '''
        check if s contains only letters 

        '''
        for c in s:
            if c  in self.sql_punct:
                return False
        return True
    
    def tokenize(self, text, return_type='int', remove_space = False):
        '''
        returns list of subword tokens
        '''
        if not text: 
            return []
        text = dedup_spaces(text)
        head = StringNode(text[0])
        curr_node = head
        merge_heap = []
        #merge_addition_set = set()
        locations = defaultdict(set)
        for i in range(1, len(text)):
            curr_node.next = StringNode(text[i], prev_node = curr_node)
            subword = curr_node.value + curr_node.next.value
            if subword in self.vocab:# and (self.vocab[subword], subword) not in merge_addition_set:
                #merge_addition_set.add((self.vocab[subword], subword))
                if subword not in locations:
                    heapq.heappush(merge_heap, (self.vocab[subword], subword))
                locations[subword].add(curr_node)
            curr_node = curr_node.next
        
        while merge_heap:
            _ , subword = heapq.heappop(merge_heap)
            nodes = locations[subword]
            del locations[subword]
            
            for node in nodes:
                if (node and node.next) and (node.can_merge and node.next.can_merge) and (node.value + node.next.value == subword):
                    self.merge_for_tokenization(node, node.next)
                    prev_node = node.prev
                    next_node = node.next
                    if prev_node and prev_node.can_merge and (prev_node.value + node.value) in self.vocab:
                        heapq.heappush(merge_heap, (self.vocab[prev_node.value + node.value], prev_node.value + node.value))
                        locations[prev_node.value + node.value].add(prev_node)
                    if next_node and next_node.can_merge and (node.value + next_node.value) in self.vocab:
                        heapq.heappush(merge_heap, (self.vocab[node.value + next_node.value], node.value + next_node.value))
                        locations[node.value + next_node.value].add(node)
                    
            
            #curr_node = head
            #self.merge_for_tokenization(subword ,merge_heap, merge_addition_set, head)
            
        
        tokens_list = []
        curr_node = head
        if return_type == 'str':
            while curr_node:
                if remove_space and curr_node.value == ' ':
                    pass
                else:
                    tokens_list.append(curr_node.value)
                curr_node = curr_node.next
        elif return_type == 'int':
            while curr_node:
                if  remove_space and curr_node.value == ' ':
                    pass
                else:
                    if curr_node.value in self.vocab:
                        tokens_list.append(self.vocab[curr_node.value])
                    else:
                        tokens_list.append(self.vocab['<UNK>'])
                curr_node = curr_node.next
            
        
        return tokens_list
    
    def merge_for_tokenization(self, left , right):
        prev_node = left.prev
        next_node = right.next
        
        left.value = left.value + right.value
        left.next = next_node
        if next_node:
            next_node.prev = left
        right.can_merge = False
    
    
    def tokenize_flash(self, text, return_type='int'):
        '''
        returns list of subword tokens
        return_type: 'int' to get token ids or 'str' to get token words
        '''
        if not text: 
            return []
        
        # split the text on non-word characters
        #text_words = re.split(r'([\W_]+)', text)
        text_words = re.findall(r' ?\w+| ?[^\w\s]+|\s+', text)
        #word_count = Counter(text_words)
        if hasattr(self, 'word_tokens_dict'):
            pass
        else:
            self.word_tokens_dict = dict()
        if hasattr(self, 'word_subwords_dict'):
            pass
        else:
            self.word_subwords_dict = dict()
        
        
        for word in text_words:
            # if word already tokenized , add tokens to final token_list
            if word in self.word_tokens_dict:
                #token_list.extend(word_tokens_dict[word])
                continue
            
            if len(word) == 1:
                if word in self.vocab:
                    self.word_tokens_dict[word] = [self.vocab[word]]
                else:
                    self.word_tokens_dict[word] = [self.vocab['<UNK>']]
                self.word_subwords_dict[word] = word 
                continue
            
            
            # split the word on characters and count consecutive pairs
            # create heap for possible merges
            
            pair_heap = []
            curr_split = list(word)
            curr_pairs = set()
            
            for i in range(len(word)-1):
                curr_pairs.add((word[i], word[i+1]))
                if word[i]+word[i+1] in self.vocab:
                    heapq.heappush(pair_heap, (self.vocab[word[i]+word[i+1]], (word[i], word[i+1])))   #heap updated
             
            # pop from pair_heap till its empty
            while pair_heap:
                merge_pair = None
                while pair_heap:
                    vocab_idx, (left, right) = heapq.heappop(pair_heap)
                    # skpit untill this pair is there in curr_pairs
                    if (left, right) in curr_pairs:
                        merge_pair = (left, right)
                        break
                if merge_pair is None:
                    continue
                
                new_split = []
                new_pairs = set()
                new_pair_heap = []
                i = 0
                
                # apply merge
                while i < len(curr_split):
                    if i < len(curr_split)-1 and curr_split[i] == left and curr_split[i+1] == right:
                        new_split.append(left+right)
                        if len(new_split) >= 2:
                            new_pairs.add((new_split[-2], new_split[-1]))
                            if new_split[-2]+new_split[-1] in self.vocab:
                                heapq.heappush(new_pair_heap, (self.vocab[new_split[-2]+new_split[-1]], (new_split[-2],new_split[-1])))
                        i += 2
                        continue 
                    new_split.append(curr_split[i])
                    if len(new_split) >= 2:
                        new_pairs.add((new_split[-2], new_split[-1]))
                        if new_split[-2]+new_split[-1] in self.vocab:
                            heapq.heappush(new_pair_heap, (self.vocab[new_split[-2]+new_split[-1]], (new_split[-2],new_split[-1])))
                    i += 1
                
                # update curr_pairs , curr_split, pair_heap
                curr_pairs = new_pairs 
                curr_split = new_split 
                pair_heap = new_pair_heap 
            
            curr_word_tokens = []
            for sub_word in curr_split:
                if sub_word in self.vocab:
                    curr_word_tokens.append(self.vocab[sub_word])
                else:
                    curr_word_tokens.append(self.vocab['<UNK>'])
            
            self.word_tokens_dict[word] = curr_word_tokens 
            self.word_subwords_dict[word] = curr_split
        
            
        if return_type == 'int':
            token_list = []
            for word in text_words:
                token_list.extend(self.word_tokens_dict[word])
            return token_list
        elif return_type == 'str':
            subword_list = []
            for word in text_words:
                subword_list.extend(self.word_subwords_dict[word])
            return subword_list 
        else:
            raise ValueError(f"Invalid format. format has to be either 'int' or 'str' but got '{format}'")

  
    
    def decode(self, ids):
        subword_list = []
        
        for idx in ids:
            subword_list.append(self.rev_vocab[idx])
        
        return subword_list
            
        
    

if __name__ == '__main__':
    
    text = nltk.corpus.gutenberg.raw('carroll-alice.txt')
    
    bpe = BPE(30000)
    
    start_time = time.perf_counter()
    bpe.train(text)
    end_time = time.perf_counter()
    
    execution_time = end_time - start_time
    print(f"Train time for my BPE: {execution_time:.4f} seconds")
    
    
        
