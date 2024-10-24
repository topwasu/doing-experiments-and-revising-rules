import asyncio
import re
import logging
import numpy as np
from multiprocessing import Pool

from toptoolkit.llm import create_llm


log = logging.getLogger(__name__)


def add_quotation(words):
    return ["'" + word + "'" for word in words]


class LLMScientistLDA:
    def __init__(self, config):
        self.score_method = config.agent.score_method
        self.active_method = config.agent.active_method
        self.hypotheses_method = config.aegnt.hypotheses_method
        self.start_size = config.agent.start_size
        self.word_batch_size = config.agent.word_batch_size
        self.online_it = config.agent.online_it
        self.num_active_per_it = config.agent.num_active_per_it
        self.num_eval = config.agent.num_eval
        self.llm = create_llm('gpt-3.5-turbo')

        with open('prompts/pre_hypothesis2.txt') as f:
            self.pre_hypothesis_prompt = f.read()
    
    def _get_hypothesis_from_pos_neg(self, words, labels):
        prompt = self.pre_hypothesis_prompt
        # prompt += f"Positive list (related words): {list(words[labels == True])}\n"
        # prompt += f"Negative list (non-related words): {list(words[labels == False])}\n"
        prompt += f"Related words: {list(words[labels == True])}\n"
        prompt += f"Unrelated words: {list(words[labels == False])}\n"
        prompt += "Please give your answer in the format 'Topic : '. Do not print anything else."

        response = self.llm.prompt([prompt], temperature=0)[0]

        splitted = re.split(': |\n', response)
        clean_hypotheses = []
        for x in splitted:
            if not (x.startswith('Topic') or len(x) == 0):
                clean_hypotheses.append(x)

        return clean_hypotheses
    
    def _get_sub_hypotheses_one(self, words, labels):
        prompt = self.pre_hypothesis_prompt
        # prompt += f"Positive list (related words): {list(words[labels == True])}\n"
        # prompt += f"Negative list (non-related words): {list(words[labels == False])}\n"
        prompt += f"Related words: {list(words[labels == True])}\n"
        prompt += f"Unrelated words: {list(words[labels == False])}\n"
        prompt += "Please generate 20 possible topics in the format 'Topic 1: ', 'Topic 2: ', ...., 'Topic 20: '. Do not print anything else."

        response = self.llm.prompt([prompt], temperature=0)[0]

        splitted = re.split(': |\n', response)
        clean_hypotheses = []
        for x in splitted:
            if not (x.startswith('Topic') or len(x) == 0):
                clean_hypotheses.append(x)

        return clean_hypotheses
    
    def _get_sub_hypotheses_many(self, words, labels):
        prompt = self.pre_hypothesis_prompt
        # prompt += f"Positive list (related words): {list(words[labels == True])}\n"
        # prompt += f"Negative list (non-related words): {list(words[labels == False])}\n"
        prompt += f"Related words: {list(words[labels == True])}\n"
        prompt += f"Unrelated words: {list(words[labels == False])}\n"
        prompt += "Please answer in the format 'Topic: '. Do not print anything else."

        raise NotImplementedError
        # messages = [
        #     {"role": "user", "content": prompt}
        # ]
        # responses = prompt_chatgpt(messages, n=20, temperature=1.5)

        hypotheses = []
        for response in responses:
            splitted = re.split(': |\n', response)
            for x in splitted:
                if not (x.startswith('Topic') or len(x) == 0):
                    hypotheses.append(x)

        return hypotheses
    
    def _get_sub_hypotheses(self, words, labels):
        if self.hypotheses_method == 'one':
            return self._get_sub_hypotheses_one(words, labels)
        elif self.hypotheses_method == 'many':
            return self._get_sub_hypotheses_many(words, labels)
        else:
            raise NotImplementedError
    
    def _get_a_committee(self, query_set):
        committee = []
        for _ in range(self.num_committee):
            l = np.random.randint(len(query_set) // 2, len(query_set))
            committee.append(np.random.choice(query_set, l, replace=False))
        return committee
    
    def _choose_words_by_hypothesis(self, hypothesis, words, num_choose=20, ct=0):
        if ct >= 5:
            return []
        
        prompt = f"Pick exactly {num_choose} words out of the following list of word that best belong to the topic '{hypothesis}'\n"\
            + 'The list of words is:\n'

        for idx, word in enumerate(words):
            prompt += f"#{idx + 1}: {word}\n"

        prompt += "\n Put your answer in the format '1. #id: ', '2. #id: ' and so on\n" 

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
        if len(res) < 5:
            log.info(f'Bad response -- retrying for the {ct + 1}th time')
            return self._choose_words_by_hypothesis(hypothesis, words, num_choose=num_choose, ct=ct+1)
        log.debug(f'Out of {words}, the chosen words are {res}')
        return res
    
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
    
    def _score_words_by_hypothesis_yn(self, hypothesis, words):
        log.debug(f'Scoring "{hypothesis}" using yn method...')
        prompts = []
        prompt_fst = f"Given the topic '{hypothesis}', please answer either yes or no (Do NOT answer maybe or possibly) on whether each of the following word is relevant to the topic.\n"\
            + f"The words are as follows\n"
        prompt_lst = "Give the yes/no answer in the format '#id: word, answer'\n"
        for ind in range(0, len(words), self.word_batch_size):
            batched_words = words[ind: ind+self.word_batch_size]
            batched_words_prompt = ""
            for idx, word in enumerate(batched_words):
                batched_words_prompt += f"#{idx + 1}: {word}\n"
            prompts.append(prompt_fst + batched_words_prompt + prompt_lst)

        responses = self.llm.prompt(prompts, temperature=0)

        votes = self._collect_async_responses_to_votes(responses, words)
        if len(np.nonzero(votes)[0]) < self.word_batch_size // 10: 
            log.warn(f'Scoring {hypothesis} -- gets {len(votes) - len(np.nonzero(votes)[0])} bad responses')
            log.warn(responses)
        return votes
    
    def _score_words_by_hypothesis_10(self, hypothesis, words):
        log.debug(f'Scoring "{hypothesis}" using 10 method...')
        prompts = []
        prompt_fst = f"Given the topic '{hypothesis}', please rate on a scale of 1-10 on how much each of the following word is relevant to the topic.\n"\
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
            log.warn(f'Scoring {hypothesis} -- gets {len(votes) - len(np.nonzero(votes)[0])} bad responses')
            log.warn(responses)

        # Turn rating into yes and no
        new_votes = votes.copy()
        new_votes = (votes - 5.5) / 4.5
        new_votes[votes == 0] = 0
        return new_votes
    
    def _score_words_by_hypothesis_score(self, hypothesis, words):
        log.info(f'Scoring "{hypothesis}" using score method...')
        rel_prompts = []
        unrel_prompts = []
        for word in words: # Can optimize this
            rel_prompts.append(f"The word '{word}' and the topic '{hypothesis}' are related")
            unrel_prompts.append(f"The word '{word}' and the topic '{hypothesis}' are unrelated")
        
        raise NotImplementedError
        # rel_prompt_logprobs = get_gpt_logprobs(rel_prompts)
        # unrel_prompt_logprobs = get_gpt_logprobs(unrel_prompts)

        scores = []
        for rel_logprobs, unrel_logprobs in zip(rel_prompt_logprobs, unrel_prompt_logprobs):
            rel_logprob = rel_logprobs[-1]
            unrel_logprob = unrel_logprobs[-1]
            normalized_rel_prob =  np.exp(rel_logprob) / (np.exp(rel_logprob) + np.exp(unrel_logprob))
            scores.append(normalized_rel_prob)
        return np.asarray(scores)
    
    def _score_words_by_hypothesis(self, hypothesis, words):
        if self.score_method == 'yn':
            return self._score_words_by_hypothesis_yn(hypothesis, words)
        elif self.score_method == '10':
            return self._score_words_by_hypothesis_10(hypothesis, words)
        elif self.score_method == 'score':
            return self._score_words_by_hypothesis_score(hypothesis, words)
        else:
            raise NotImplementedError
    
    def _score_current_set(self, current_set, current_labels, predict_set):
        scores = np.zeros(len(predict_set))
        sub_hypotheses = self._get_sub_hypotheses(current_set, current_labels)
        log.info(f'Sub hypotheses: {sub_hypotheses}')
        indices_list = [np.random.permutation(len(predict_set)) for _ in sub_hypotheses]
        args_list = [(hypothesis, predict_set[indices]) for indices, hypothesis in zip(indices_list, sub_hypotheses)]
        # args_list = [(hypothesis, predict_set for hypothesis in sub_hypotheses]
        with Pool(20) as p:
            scores = p.starmap(self._score_words_by_hypothesis, args_list)
        for i, indices in enumerate(indices_list):
            new_score = np.zeros(len(predict_set))
            for old_idx, idx in enumerate(indices):
                new_score[idx] = scores[i][old_idx]
            scores[i] = new_score

        log.info(f'All scores {scores}')            
        scores = np.sum(scores, axis=0) / len(sub_hypotheses)
        log.info(f'Computed scores {scores}')

        return scores

    def evaluate(self, test_dataloader, ind2word):
        results = []
        all_words = np.asarray(list(ind2word.values()))
        
        for ex_id, x in enumerate(test_dataloader):
            ex_res = []
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
            ex_res.append(res)

            for it in range(self.online_it):
                log.info(f'Example {ex_id} Iteration {it} - START')
                log.info(f'Example {ex_id} Iteration {it} Current set: {current_set}')
                log.info(f'Example {ex_id} Iteration {it} Current labels: {current_labels}')

                # Get candidate set - everything in vocab that's not in current set
                candidate_indices = []
                for idx, word in enumerate(all_words):
                    if word not in current_set:
                        candidate_indices.append(idx)
                candidate_indices = np.asarray(candidate_indices)
                candidate_set = all_words[candidate_indices]

                # TODO: Fix this
                log.info("Choosing words to query...")
                if self.active_method == 'active':
                    labeled_words = self._query_active(current_scores, candidate_set, all_words)
                elif self.active_method == 'active_committee':
                    raise NotImplementedError
                    # labeled_words = self._query_active_committee(current_set, current_labels, candidate_set, all_words, pos_words) # TODO: FIX this
                elif self.active_method == 'random':
                    labeled_words = self._query_random(candidate_set)
                elif self.active_method == 'predict':
                    labeled_words = self._query_predict(current_scores, candidate_set, all_words)
                else:
                    raise NotImplementedError

                labels = np.asarray([True if word in pos_words else False for word in labeled_words])
                log.info(f'Labeling the following words: {labeled_words}')
                log.info(f'Labels: {labels}')

                current_set = np.concatenate((current_set, labeled_words), axis=0)
                current_labels = np.concatenate((current_labels, labels), axis=0)

                current_scores = self._score_current_set(current_set, current_labels, all_words)
                chosen_indices = np.argsort(current_scores)[-self.num_eval:]
                chosen_words = all_words[chosen_indices]
                chosen_scores = current_scores[chosen_indices]
                res = sum([1 if word in pos_words else 0 for word in chosen_words])
                log.info(f'Example {ex_id} Iteration {it} Chosen words: {chosen_words}')
                log.info(f'Example {ex_id} Iteration {it} Chosen scores: {chosen_scores}')
                log.info(f'Example {ex_id} Iteration {it} Results: {res}/{self.num_eval}')
                ex_res.append(res)
            results.append(ex_res)
        return np.asarray(results)
    
    def _query_active(self, current_scores, candidate_set, all_words):
        uncertainty_rank = np.argsort(np.abs(current_scores))
        res = []
        for idx in uncertainty_rank:
            if all_words[idx] in candidate_set:
                res.append(all_words[idx])
                if len(res) == self.num_active_per_it:
                    break
        res = np.asarray(res)
        return res
    
    def _query_active_committee(self, current_set, current_labels, candidate_set, all_words, pos_words):
        # Get sub-hypotheses
        sub_hypotheses = self._get_sub_hypotheses(current_set, current_labels)
        log.info('Sub hypotheses: ')
        log.info(sub_hypotheses)

        # Get votes 
        log.info("Getting votes...")
        votes = self._get_votes_from_hypotheses(sub_hypotheses, all_words)
        log.debug(f'Votes: {votes}')
        certainty = []
        for word_idx in range(votes.shape[1]):
            word_votes = votes[:, word_idx]
            certainty.append(np.mean(word_votes[word_votes != 0])) # Low means very uncertain
        log.debug(f'Certainty: {certainty}')

        most_voted_words = all_words[np.argsort(certainty)[-20:]]
        log.info(f'Committee Results: {sum([1 if word in pos_words else 0 for word in most_voted_words])}/20')

        # Rank uncertainty based votes
        # uncertainty_rank = np.argsort(np.abs(percentage - 0.5))
        uncertainty_rank = np.argsort(np.abs(certainty))

        # Obtain label for those
        # labeled_words = candidate_set[uncertainty_rank[:self.num_active_per_it]]
        labeled_words = []
        for rank in uncertainty_rank:
            if all_words[rank] in candidate_set:
                labeled_words.append(all_words[rank])
                if len(labeled_words) >= self.num_active_per_it:
                    break
        return labeled_words
    
    def _query_random(self, candidate_set):
        return candidate_set[np.random.choice(len(candidate_set), self.num_active_per_it, replace=False)]
    
    def _query_predict(self, current_scores, candidate_set, all_words):
        uncertainty_rank = np.argsort(-current_scores)
        res = []
        for idx in uncertainty_rank:
            if all_words[idx] in candidate_set:
                res.append(all_words[idx])
                if len(res) == self.num_active_per_it:
                    break
        res = np.asarray(res)
        return res
