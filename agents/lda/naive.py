import asyncio
import re
import logging
import numpy as np
from multiprocessing import Pool

from toptoolkit.llm import create_llm


log = logging.getLogger(__name__)

class LLMNaiveLDA():
    def __init__(self, config):
        self.score_method = config.score_method
        self.start_size = config.start_size
        self.word_batch_size = config.word_batch_size
        self.num_eval = config.num_eval
        self.llm = create_llm('gpt-3.5-turbo')

    def _collect_async_responses_to_votes(self, responses, words):
        votes = []
        num_word_batches = -(-len(words)//self.word_batch_size)
        for idx, response in enumerate(responses):
            st_ind = (idx % num_word_batches) * self.word_batch_size
            ind = st_ind
            splitted = re.split(': |\n', response)
            for x in splitted:
                if not (x.startswith('#') or len(x) == 0):
                    try:
                        word, yn = x.split(', ')
                        if word in words[ind:st_ind+self.word_batch_size]:
                            while word != words[ind]:
                                votes.append(0)
                                ind += 1
                            if yn in ['1' , '2', '3', '4', '5', '6', '7', '8', '9', '10']:
                                votes.append(int(yn))
                            else:
                                votes.append(1 if yn.lower() == 'yes' else (-1 if yn.lower() == 'no' else 0))
                            ind += 1
                    except:
                        continue
            while ind < min(st_ind + self.word_batch_size, len(words)):
                votes.append(0)
                ind +=1
        votes = np.asarray(votes)
        return votes
    
    def _choose_words_from_set(self, current_set, words):
        prompt = f"Given the set of words {list(current_set)}, pick exactly 50 words out of the following list of word that are most relevant to the given set of words'\n"\
            + 'The list of words is:\n'

        for idx, word in enumerate(words):
            prompt += f"{word}\n"

        prompt += "\nPut your answer in the format '1. ', '2. ', ..., '50. '\n" 

        response = self.llm.prompt([prompt], temperature=0)[0]

        splitted = re.split(r'\n', response)
        clean_answers = []
        for x in splitted:
            try:
                clean_answers.append(x.split(': ')[-1].split('#')[-1].split('. ')[-1].lower().strip())
            except:
                continue
        
        res = []
        for a in clean_answers:
            if a in words:
                res.append(a)
        return res

    def _score_current_set_choose(self, current_set, current_labels, predict_set):
        words = predict_set
        log.info(f'Scoring "{current_set}" using choose method...')
        chosen_words = []
        batch_size = 50
        for ind in range(0, len(words), batch_size):
            log.debug(f'Choosing word - progress: {ind}/{len(words)}')
            chosen_words = self._choose_words_from_set(current_set, list(words[ind: ind+batch_size]) + chosen_words)
        
        votes = []
        for word in predict_set:
            if word in chosen_words:
                votes.append(1)
            else:
                votes.append(-1)
        votes = np.asarray(votes)
        return votes

    def _score_current_set_10(self, current_set, current_labels, predict_set):
        words = predict_set
        log.info(f'Scoring "{current_set}" using 10 method...')
        prompts = []
        prompt_fst = f"Given the set of words {list(current_set)}, please rate on a scale of 1-10 on how much each of the following word is relevant to the given set of words.\n"\
            + f"The words are as follows\n"
        prompt_lst = "Give the rating in the format '#id: word, rating'\n"
        for ind in range(0, len(words), self.word_batch_size):
            batched_words = words[ind: ind+self.word_batch_size]
            batched_words_prompt = ""
            for idx, word in enumerate(batched_words):
                batched_words_prompt += f"#{idx + 1}: {word}\n"
            prompts.append(prompt_fst + batched_words_prompt + prompt_lst)

        responses = self.llm.prompt(prompts, temperature=0)

        votes = self._collect_async_responses_to_votes(responses, words)
        if len(np.nonzero(votes)[0]) < self.word_batch_size // 10: 
            log.warn(f'Scoring {current_set} -- gets {len(votes) - len(np.nonzero(votes)[0])} bad responses')
            log.warn(responses)

        # Turn rating into yes and no
        new_votes = votes.copy()
        new_votes = (votes - 5.5) / 4.5
        new_votes[votes == 0] = 0
        return new_votes

    def _score_current_set(self, current_set, current_labels, predict_set):
        if self.score_method == 'choose':
            return self._score_current_set_choose(current_set, current_labels, predict_set)
        elif self.score_method == '10':
            return self._score_current_set_10(current_set, current_labels, predict_set)
        else:
            raise NotImplementedError

    def evaluate(self, test_dataloader, ind2word):
        results = []
        all_words = np.asarray(list(ind2word.values()))
        
        for ex_id, x in enumerate(test_dataloader):
            x = x[0].numpy()
            pos_words = np.asarray([ind2word[ind] for ind in x])
            log.info(f'The positive words of this concept: {pos_words}')

            current_set = np.asarray(pos_words[:self.start_size])
            current_labels = np.asarray([True] * self.start_size)

            # Get hypothesis
            candidate_set = np.asarray(list(filter(lambda x: x not in current_set, all_words)))
            # current_hypothesis = self._get_hypothesis_from_pos_neg(current_set, current_labels)[-1]

            current_scores = self._score_current_set(current_set, current_labels, all_words)
            chosen_indices = np.argsort(current_scores)[-self.num_eval:]
            chosen_words = all_words[chosen_indices]
            chosen_scores = current_scores[chosen_indices]
            res = sum([1 if word in pos_words else 0 for word in chosen_words])
            log.info(f'Example {ex_id} Iteration -1 Chosen words: {chosen_words}')
            log.info(f'Example {ex_id} Iteration -1 Chosen scores: {chosen_scores}')
            log.info(f'Example {ex_id} Iteration -1 Results: {res}/{self.num_eval}')
            results.append(res)
        return results