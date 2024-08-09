import os
import os.path
import json
import numpy as np
from itertools import accumulate
import sys
from typing import List, Tuple
from sklearn.metrics import precision_recall_fscore_support as prfs

METRIC_LABELS = ['prec_micro', 'rec_micro', 'f1_micro', 'prec_macro', 'rec_macro', 'f1_macro']

rel2id = json.load(open('meta/rel2id.json', 'r'))
id2rel = {value: key for key, value in rel2id.items()}




def findSmallestDifference(A, B, m, n):
 
    # Sort both arrays
    # using sort function
    A.sort()
    B.sort()
 
    a = 0
    b = 0
 
    # Initialize result as max value
    result = sys.maxsize
 
    # Scan Both Arrays upto
    # sizeof of the Arrays
    while (a < m and b < n):
     
        if (abs(A[a] - B[b]) < result):
            result = abs(A[a] - B[b])
 
        # Move Smaller Value
        if (A[a] < B[b]):
            a += 1
 
        else:
            b += 1
    # return final sma result
    return result




def to_official(preds, features):
    h_idx, t_idx, title = [], [], []

    for f in features:
        hts = f["hts"]
        h_idx += [ht[0] for ht in hts]
        t_idx += [ht[1] for ht in hts]
        title += [f["title"] for ht in hts]

    res = []
    for i in range(preds.shape[0]):
        pred = preds[i]
        pred = np.nonzero(pred)[0].tolist()
        for p in pred:
            if p != 0 and p < 97:
                res.append(
                    {
                        'title': title[i],
                        'h_idx': h_idx[i],
                        't_idx': t_idx[i],
                        'r': id2rel[p],
                    }
                )
    return res

def to_official_by_doc(preds, features):
    h_idx, t_idx, title = [], [], []
    res = []
    for pred, f in zip(preds, features):
        hts = f["hts"]
        h_idx = [ht[0] for ht in hts]
        t_idx = [ht[1] for ht in hts]
        title = [f["title"] for ht in hts]
        local_res = []
        

        for i in range(pred.shape[0]):
            pred_i = np.nonzero(pred[i])[0].tolist()
            for p in pred_i:
                if p != 0 and p < 97:
                    local_res.append(
                        {
                            'title': title[i],
                            'h_idx': h_idx[i],
                            't_idx': t_idx[i],
                            'r': id2rel[p],
                        }
                    )
        res.append(local_res)
    return res

def gen_train_facts(data_file_name, truth_dir):
    fact_file_name = data_file_name[data_file_name.find("train_"):]
    fact_file_name = os.path.join(truth_dir, fact_file_name.replace(".json", ".fact"))

    if os.path.exists(fact_file_name):
        fact_in_train = set([])
        triples = json.load(open(fact_file_name))
        for x in triples:
            fact_in_train.add(tuple(x))
        return fact_in_train

    fact_in_train = set([])
    ori_data = json.load(open(data_file_name))
    for data in ori_data:
        vertexSet = data['vertexSet']
        for label in data['labels']:
            rel = label['r']
            for n1 in vertexSet[label['h']]:
                for n2 in vertexSet[label['t']]:
                    fact_in_train.add((n1['name'], n2['name'], rel))

    json.dump(list(fact_in_train), open(fact_file_name, "w"))

    return fact_in_train


def official_evaluate(tmp, path, train_file, dev_file):
    '''
        Adapted from the official evaluation code
    '''
    truth_dir = os.path.join(path, 'ref')

    if not os.path.exists(truth_dir):
        os.makedirs(truth_dir)

    fact_in_train_annotated = gen_train_facts(os.path.join(path, train_file), truth_dir)
    fact_in_train_distant = gen_train_facts(os.path.join(path, "train_distant.json"), truth_dir)

    truth = json.load(open(os.path.join(path, dev_file)))

    std = {}
    tot_evidences = 1
    titleset = set([])

    title2vectexSet = {}

    for x in truth:
        title = x['title']
        titleset.add(title)

        vertexSet = x['vertexSet']
        title2vectexSet[title] = vertexSet

        for label in x['labels']:
            r = label['r']
            h_idx = label['h']
            t_idx = label['t']
            std[(title, r, h_idx, t_idx)] = set(label['evidence'])
            tot_evidences += len(label['evidence'])

    tot_relations = len(std)
    tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
    if len(tmp)>0:
        submission_answer = [tmp[0]]
    else:
        return 0, 0, 0, 0 , 0, 0
    for i in range(1, len(tmp)):
        x = tmp[i]
        y = tmp[i - 1]
        if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
            submission_answer.append(tmp[i])

    correct_re = 0
    correct_evidence = 0
    pred_evi = 0

    correct_in_train_annotated = 0
    correct_in_train_distant = 0
    titleset2 = set([])
    for x in submission_answer:
        title = x['title']
        h_idx = x['h_idx']
        t_idx = x['t_idx']
        r = x['r']
        titleset2.add(title)
        if title not in title2vectexSet:
            continue
        vertexSet = title2vectexSet[title]

        if 'evidence' in x:
            evi = set(x['evidence'])
        else:
            evi = set([])
        pred_evi += len(evi)

        if (title, r, h_idx, t_idx) in std:
            correct_re += 1
            stdevi = std[(title, r, h_idx, t_idx)]
            correct_evidence += len(stdevi & evi)
            in_train_annotated = in_train_distant = False
            for n1 in vertexSet[h_idx]:
                for n2 in vertexSet[t_idx]:
                    if (n1['name'], n2['name'], r) in fact_in_train_annotated:
                        in_train_annotated = True
                    if (n1['name'], n2['name'], r) in fact_in_train_distant:
                        in_train_distant = True

            if in_train_annotated:
                correct_in_train_annotated += 1
            if in_train_distant:
                correct_in_train_distant += 1

    re_p = 1.0 * correct_re / len(submission_answer)
    re_r = 1.0 * correct_re / tot_relations
    if re_p + re_r == 0:
        re_f1 = 0
    else:
        re_f1 = 2.0 * re_p * re_r / (re_p + re_r)

    evi_p = 1.0 * correct_evidence / pred_evi if pred_evi > 0 else 0
    evi_r = 1.0 * correct_evidence / tot_evidences
    if evi_p + evi_r == 0:
        evi_f1 = 0
    else:
        evi_f1 = 2.0 * evi_p * evi_r / (evi_p + evi_r)

    re_p_ignore_train_annotated = 1.0 * (correct_re - correct_in_train_annotated) / (len(submission_answer) - correct_in_train_annotated + 1e-5)
    re_p_ignore_train = 1.0 * (correct_re - correct_in_train_distant) / (len(submission_answer) - correct_in_train_distant + 1e-5)

    if re_p_ignore_train_annotated + re_r == 0:
        re_f1_ignore_train_annotated = 0
    else:
        re_f1_ignore_train_annotated = 2.0 * re_p_ignore_train_annotated * re_r / (re_p_ignore_train_annotated + re_r)

    if re_p_ignore_train + re_r == 0:
        re_f1_ignore_train = 0
    else:
        re_f1_ignore_train = 2.0 * re_p_ignore_train * re_r / (re_p_ignore_train + re_r)

    return re_f1, evi_f1, re_f1_ignore_train_annotated, re_f1_ignore_train , re_p, re_r



def official_evaluate_benchmark(tmp, path, train_file, dev_file):
    '''
        Adapted from the official evaluation code
    '''
    freq_keys = set(['P17', 'P131', 'P27', 'P150', 'P175', 'P577', 'P463', 'P527', 'P495', 'P361'])
    long_tail_keys = set(rel2id.keys()) - freq_keys
    truth_dir = os.path.join(path, 'ref')

    if not os.path.exists(truth_dir):
        os.makedirs(truth_dir)

    fact_in_train_annotated = gen_train_facts(os.path.join(path, train_file), truth_dir)
    fact_in_train_distant = gen_train_facts(os.path.join(path, "train_distant.json"), truth_dir)

    truth = json.load(open(os.path.join(path, dev_file)))

    std = {}
    std_freq = {}
    std_long_tail = {}
    tot_evidences = 1
    titleset = set([])

    title2vectexSet = {}
    std_intra = {}
    std_inter = {}
    std_inter_long = {}
    for x in truth:
        title = x['title']
        titleset.add(title)

        vertexSet = x['vertexSet']
        title2vectexSet[title] = vertexSet

        for label in x['labels']:
            r = label['r']
            h_idx = label['h']
            t_idx = label['t']
            h_sent_set = [x['sent_id'] for x in vertexSet[h_idx]]
            t_sent_set = [x['sent_id'] for x in vertexSet[t_idx]]
            
            std[(title, r, h_idx, t_idx)] = set(label['evidence'])
            tot_evidences += len(label['evidence'])
            if findSmallestDifference(h_sent_set, t_sent_set, len(h_sent_set),len(t_sent_set) )==0:
                std_intra[(title, r, h_idx, t_idx)] = set(label['evidence'])
            if 1 <= findSmallestDifference(h_sent_set, t_sent_set, len(h_sent_set),len(t_sent_set) )  :
                std_inter[(title, r, h_idx, t_idx)] = set(label['evidence'])
            if 5 < findSmallestDifference(h_sent_set, t_sent_set, len(h_sent_set),len(t_sent_set) )  :
                std_inter_long[(title, r, h_idx, t_idx)] = set(label['evidence'])
            if r in freq_keys:
                std_freq[(title, r, h_idx, t_idx)] = set(label['evidence'])
            if r in long_tail_keys:
                std_long_tail[(title, r, h_idx, t_idx)] = set(label['evidence'])
                
    tot_relations = len(std)
    tot_relations_freq = len(std_freq)
    tot_relations_long_tail = len(std_long_tail)
    tot_relations_intra = len(std_intra)
    tot_relations_inter = len(std_inter)
    tot_relations_inter_long = len(std_inter_long)
    
    tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
    if len(tmp) > 1:
        submission_answer = [tmp[0]]
        for i in range(1, len(tmp)):
            x = tmp[i]
            y = tmp[i - 1]
            if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
                submission_answer.append(tmp[i]) 
    else: 
        submission_answer = []
    submission_answer_freq = []
    submission_answer_long_tail =[] 

    submission_answer_freq = [x for x in submission_answer if x['r'] in freq_keys]
    submission_answer_long_tail = [x for x in submission_answer if x['r'] in long_tail_keys]
    submission_answer_intra = []
    submission_answer_inter = []
    submission_answer_inter_long = []
    for i in range(len(submission_answer)):
        vertexSet = title2vectexSet[submission_answer[i]['title']] 
        if title not in title2vectexSet:
            print(title)
            continue
        h_sent_set = [x['sent_id'] for x in vertexSet[submission_answer[i]['h_idx']]]
        t_sent_set = [x['sent_id'] for x in vertexSet[submission_answer[i]['t_idx']]]
        if findSmallestDifference(h_sent_set, t_sent_set, len(h_sent_set),len(t_sent_set) )==0:
            submission_answer_intra.append(submission_answer[i])
        if 1<= findSmallestDifference(h_sent_set, t_sent_set, len(h_sent_set),len(t_sent_set))  :
            submission_answer_inter.append(submission_answer[i])
        if 5 < findSmallestDifference(h_sent_set, t_sent_set, len(h_sent_set),len(t_sent_set)) :
            submission_answer_inter_long.append(submission_answer[i])

    correct_re = 0
    correct_re_freq = 0
    correct_re_long_tail = 0
    correct_re_intra = 0
    correct_re_inter = 0
    correct_re_inter_long = 0
    correct_evidence = 0
    pred_evi = 0

    correct_in_train_annotated = 0
    correct_in_train_distant = 0
    titleset2 = set([])
    for x in submission_answer:
        title = x['title']
        h_idx = x['h_idx']
        t_idx = x['t_idx']
        r = x['r']
        titleset2.add(title)
        if title not in title2vectexSet:
            continue
        vertexSet = title2vectexSet[title]

        if 'evidence' in x:
            evi = set(x['evidence'])
        else:
            evi = set([])
        pred_evi += len(evi)

        if (title, r, h_idx, t_idx) in std:
            correct_re += 1
            stdevi = std[(title, r, h_idx, t_idx)]
            correct_evidence += len(stdevi & evi)
            in_train_annotated = in_train_distant = False
            for n1 in vertexSet[h_idx]:
                for n2 in vertexSet[t_idx]:
                    if (n1['name'], n2['name'], r) in fact_in_train_annotated:
                        in_train_annotated = True
                    if (n1['name'], n2['name'], r) in fact_in_train_distant:
                        in_train_distant = True

            if in_train_annotated:
                correct_in_train_annotated += 1
            if in_train_distant:
                correct_in_train_distant += 1
    for x in submission_answer_freq:
        title = x['title']
        h_idx = x['h_idx']
        t_idx = x['t_idx']
        r = x['r']
        titleset2.add(title)
        if title not in title2vectexSet:
            continue
        vertexSet = title2vectexSet[title]

        if (title, r, h_idx, t_idx) in std_freq:
            correct_re_freq += 1
    for x in submission_answer_long_tail:
        title = x['title']
        h_idx = x['h_idx']
        t_idx = x['t_idx']
        r = x['r']
        titleset2.add(title)
        if title not in title2vectexSet:
            continue
        vertexSet = title2vectexSet[title]

        if (title, r, h_idx, t_idx) in std_long_tail:
            correct_re_long_tail += 1

    for x in submission_answer_intra:
        title = x['title']
        h_idx = x['h_idx']
        t_idx = x['t_idx']
        r = x['r']
        titleset2.add(title)
        if title not in title2vectexSet:
            continue
        vertexSet = title2vectexSet[title]

        if (title, r, h_idx, t_idx) in std_intra:
            correct_re_intra += 1
    for x in submission_answer_inter:
        title = x['title']
        h_idx = x['h_idx']
        t_idx = x['t_idx']
        r = x['r']
        titleset2.add(title)
        if title not in title2vectexSet:
            continue
        vertexSet = title2vectexSet[title]

        if (title, r, h_idx, t_idx) in std_inter:
            correct_re_inter += 1

 
    for x in submission_answer_inter_long:
        title = x['title']
        h_idx = x['h_idx']
        t_idx = x['t_idx']
        r = x['r']
        titleset2.add(title)
        if title not in title2vectexSet:
            continue
        vertexSet = title2vectexSet[title]

        if (title, r, h_idx, t_idx) in std_inter_long:
            correct_re_inter_long += 1


    if len(submission_answer)>0:        
        re_p = 1.0 * correct_re / len(submission_answer)
    else:
        re_p = 0
    re_r = 1.0 * correct_re / tot_relations
    if re_p + re_r == 0:
        re_f1 = 0
    else:
        re_f1 = 2.0 * re_p * re_r / (re_p + re_r)

    if len(submission_answer_freq)>0:        
        re_p_freq = 1.0 * correct_re_freq / len(submission_answer_freq)
    else:
        re_p_freq = 0
    
    re_r_freq = 1.0 * correct_re_freq / tot_relations_freq
    if re_p_freq + re_r_freq == 0:
        re_f1_freq = 0
    else:
        re_f1_freq = 2.0 * re_p_freq * re_r_freq / (re_p_freq + re_r_freq)
    if len(submission_answer_long_tail)>0:        
        re_p_long_tail = 1.0 * correct_re_long_tail / len(submission_answer_long_tail)
    else:
        re_p_long_tail = 0
    
    re_r_long_tail = 1.0 * correct_re_long_tail / tot_relations_long_tail
    if re_p_long_tail + re_r_long_tail == 0:
        re_f1_long_tail = 0
    else:
        re_f1_long_tail = 2.0 * re_p_long_tail * re_r_long_tail / (re_p_long_tail + re_r_long_tail)

    if len(submission_answer_intra)>0:        
        re_p_intra = 1.0 * correct_re_intra / len(submission_answer_intra)
    else:
        re_p_intra = 0
    
    re_r_intra = 1.0 * correct_re_intra / tot_relations_intra
    if re_p_intra + re_r_intra == 0:
        re_f1_intra = 0
    else:
        re_f1_intra = 2.0 * re_p_intra * re_r_intra / (re_p_intra + re_r_intra)

    if len(submission_answer_inter)>0:        
        re_p_inter = 1.0 * correct_re_inter / len(submission_answer_inter)
    else:
        re_p_inter = 0
    re_r_inter = 1.0 * correct_re_inter / tot_relations_inter
    if re_p_inter + re_r_inter == 0:
        re_f1_inter = 0
    else:
        re_f1_inter = 2.0 * re_p_inter * re_r_inter / (re_p_inter + re_r_inter)



    evi_p = 1.0 * correct_evidence / pred_evi if pred_evi > 0 else 0
    evi_r = 1.0 * correct_evidence / tot_evidences
    if evi_p + evi_r == 0:
        evi_f1 = 0
    else:
        evi_f1 = 2.0 * evi_p * evi_r / (evi_p + evi_r)

    re_p_ignore_train_annotated = 1.0 * (correct_re - correct_in_train_annotated) / (len(submission_answer) - correct_in_train_annotated + 1e-5)
    re_p_ignore_train = 1.0 * (correct_re - correct_in_train_distant) / (len(submission_answer) - correct_in_train_distant + 1e-5)

    if re_p_ignore_train_annotated + re_r == 0:
        re_f1_ignore_train_annotated = 0
    else:
        re_f1_ignore_train_annotated = 2.0 * re_p_ignore_train_annotated * re_r / (re_p_ignore_train_annotated + re_r)

    if re_p_ignore_train + re_r == 0:
        re_f1_ignore_train = 0
    else:
        re_f1_ignore_train = 2.0 * re_p_ignore_train * re_r / (re_p_ignore_train + re_r)

    return re_f1, evi_f1, re_f1_ignore_train_annotated, re_f1_ignore_train, re_p, re_r, re_f1_freq, re_f1_long_tail, re_f1_intra, re_f1_inter, re_p_freq, re_r_freq, re_p_long_tail, re_r_long_tail


import os
import os.path
import json
import numpy as np
from tqdm import tqdm

rel2id = json.load(open('meta/rel2id.json', 'r'))
id2rel = {value: key for key, value in rel2id.items()}


def get_title2pred(pred: list) -> dict:
    '''
    Convert predictions into dictionary.
    Input:
        :pred: list of dictionaries, each dictionary entry is a predicted relation triple. Keys: ['title', 'h_idx', 't_idx', 'r', 'evidence', 'score']
    Output:
        :title2pred: dictionary with (key, value) = (title, {rel_triple: score})
    '''

    title2pred = {}

    for p in pred:
        if p["r"] == "Na":
            continue
        curr = (p["h_idx"], p["t_idx"], p["r"])

        if p["title"] in title2pred:
            if curr in title2pred[p["title"]]:
                title2pred[p["title"]][curr] = max(p["score"], title2pred[p["title"]][curr])
            else:
                title2pred[p["title"]][curr] = p["score"]
        else:
            title2pred[p["title"]] = {curr: p["score"]}
    return title2pred


def get_title2gt(features: dict) -> dict:
    '''
    Convert ground-truth labels to dictionary.
    Input:
        :features: list of features within each document. Identical to the lists obtained from pre-processing.
    Output:
        :title2gt: dictionary with (key, value) = (title, [gold_triples])
    '''
    title2gt = {}
    for f in features:
        title = f["title"]
        title2gt[title] = []
        for idx, p in enumerate(f["hts"]):
            h, t = p
            label = np.array(f['labels'][idx])
            rs = np.nonzero(label[1:])[0] + 1  # + 1 for no-label
            title2gt[title].extend([(h, t, id2rel[r]) for r in rs])

    return title2gt


def select_thresh(cand: list, num_gt: int, correct: int, num_pred: int):
    '''
    select threshold for relation predictions.
    Input:
        :cand: list of relation candidates
        :num_gt: number of ground-truth relations.
        :correct: number of correct relation predictions selected.
        :num_pred: number of relation predictions selected.
    Output:
        :thresh: threshold for selecting relations.
        :sorted_pred: predictions selected from cand.
    '''

    sorted_pred = sorted(cand, key=lambda x: x[1], reverse=True)
    precs, recalls = [], []

    for pred in sorted_pred:
        correct += pred[0]
        num_pred += 1
        precs.append(correct / num_pred)  # Precision
        recalls.append(correct / num_gt)  # Recall

    recalls = np.asarray(recalls, dtype='float32')
    precs = np.asarray(precs, dtype='float32')
    f1_arr = (2 * recalls * precs / (recalls + precs + 1e-20))
    f1 = f1_arr.max()
    f1_pos = f1_arr.argmax()
    thresh = sorted_pred[f1_pos][1]

    print('Best thresh', thresh, '\tbest F1', f1)
    return thresh, sorted_pred[:f1_pos + 1]


def merge_results(pred: list, pred_pseudo: list, features: list, thresh: float = None):
    '''
    Merge relation predictions from the original document and psuedo documents.
    Input:
        :pred: list of dictionaries, each dictionary entry is a predicted relation triple from the original document. Keys: ['title', 'h_idx', 't_idx', 'r', 'evidence', 'score'].
        :pred_pseudo: list of dictionaries, each dictionary entry is a predicted relation triple from pseudo documents. Keys: ['title', 'h_idx', 't_idx', 'r', 'evidence', 'score'].
        :features: list of features within each document. Identical to the lists obtained from pre-processing.
        :thresh: threshold for selecting predictions.
    Output:
        :merged_res: list of merged relation predictions. Each relation prediction is a dictionay with keys (title, h_idx, t_idx, r).
        :thresh: threshold of selecting relation predictions.
    '''

    title2pred = get_title2pred(pred)
    title2pred_pseudo = get_title2pred(pred_pseudo)

    title2gt = get_title2gt(features)
    num_gt = sum([len(title2gt[t]) for t in title2gt])

    titles = list(title2pred.keys())
    cand = []
    merged_res = []
    correct, num_pred = 0, 0

    for t in titles:
        rels = title2pred[t]
        rels_pseudo = title2pred_pseudo[t] if t in title2pred_pseudo else {}

        union = set(rels.keys()) | set(rels_pseudo.keys())
        for r in union:
            if r in rels and r in rels_pseudo:  # add those into predictions
                if rels[r] > 0 and rels_pseudo[r] > 0:
                    merged_res.append({'title': t, 'h_idx': r[0], 't_idx': r[1], 'r': r[2]})
                    num_pred += 1
                    correct += r in title2gt[t]
                    continue
                score = rels[r] + rels_pseudo[r]
            elif r in rels:  # -10 for penalty
                score = rels[r] - 10
            elif r in rels_pseudo:
                score = rels_pseudo[r] - 10
            cand.append((r in title2gt[t], score, t, r[0], r[1], r[2]))

    if thresh != None:
        sorted_pred = sorted(cand, key=lambda x: x[1], reverse=True)
        last = min(filter(lambda x: x[1] > thresh, sorted_pred))
        until = sorted_pred.index(last)
        cand = sorted_pred[:until + 1]
        merged_res.extend([{'title': r[2], 'h_idx': r[3], 't_idx': r[4], 'r': r[5]} for r in cand])
        return merged_res, thresh

    if cand != []:
        thresh, cand = select_thresh(cand, num_gt, correct, num_pred)
        merged_res.extend([{'title': r[2], 'h_idx': r[3], 't_idx': r[4], 'r': r[5]} for r in cand])

    return merged_res, thresh


def extract_relative_score(scores: list, topks: list) -> list:
    '''
    Get relative score from topk predictions.
    Input:
        :scores: a list containing scores of topk predictions.
        :topks: a list containing relation labels of topk predictions.
    Output:
        :scores: a list containing relative scores of topk predictions.
    '''

    na_score = scores[-1].item() - 1
    if 0 in topks:
        na_score = scores[np.where(topks == 0)].item()

    scores -= na_score

    return scores


def to_official(preds: list, features: list, evi_preds: list = [], scores: list = [], topks: list = []):
    '''
    Convert the predictions to official format for evaluating.
    Input:
        :preds: list of dictionaries, each dictionary entry is a predicted relation triple from the original document. Keys: ['title', 'h_idx', 't_idx', 'r', 'evidence', 'score'].
        :features: list of features within each document. Identical to the lists obtained from pre-processing.
        :evi_preds: list of the evidence prediction corresponding to each relation triple prediction.
        :scores: list of scores of topk relation labels for each entity pair.
        :topks: list of topk relation labels for each entity pair.
    Output:
        :official_res: official results used for evaluation.
        :res: topk results to be dumped into file, which can be further used during fushion.
    '''

    h_idx, t_idx, title, sents = [], [], [], []

    for f in features:
        if "entity_map" in f:
            hts = [[f["entity_map"][ht[0]], f["entity_map"][ht[1]]] for ht in f["hts"]]
        else:
            hts = f["hts"]

        h_idx += [ht[0] for ht in hts]
        t_idx += [ht[1] for ht in hts]
        title += [f["title"] for ht in hts]
        sents += [len(f["sent_pos"])] * len(hts)

    official_res = []
    res = []

    for i in tqdm(range(preds.shape[0]), desc="preds"):  # for each entity pair
        if scores != []:
            score = extract_relative_score(scores[i], topks[i])
            pred = topks[i]
        else:
            pred = preds[i]
            pred = np.nonzero(pred)[0].tolist()

        for p in pred:  # for each predicted relation label (topk)
            curr_result = {
                'title': title[i],
                'h_idx': h_idx[i],
                't_idx': t_idx[i],
                'r': id2rel[p],
            }
            if evi_preds != []:
                curr_evi = evi_preds[i]
                evis = np.nonzero(curr_evi)[0].tolist()
                curr_result["evidence"] = [evi for evi in evis if evi < sents[i]]
            if scores != []:
                curr_result["score"] = score[np.where(topks[i] == p)].item()
            if p != 0 and p in np.nonzero(preds[i])[0].tolist():
                official_res.append(curr_result)
            res.append(curr_result)

    return official_res, res


def gen_train_facts(data_file_name, truth_dir):
    fact_file_name = data_file_name[data_file_name.find("train_"):]
    fact_file_name = os.path.join(truth_dir, fact_file_name.replace(".json", ".fact"))

    if os.path.exists(fact_file_name):
        fact_in_train = set([])
        triples = json.load(open(fact_file_name))
        for x in triples:
            fact_in_train.add(tuple(x))
        return fact_in_train

    fact_in_train = set([])
    ori_data = json.load(open(data_file_name))
    for data in ori_data:
        vertexSet = data['vertexSet']
        for label in data['labels']:
            rel = label['r']
            for n1 in vertexSet[label['h']]:
                for n2 in vertexSet[label['t']]:
                    fact_in_train.add((n1['name'], n2['name'], rel))

    json.dump(list(fact_in_train), open(fact_file_name, "w"))

    return fact_in_train


def official_evaluate(tmp, path, train_file="train_annotated.json", dev_file="dev.json"):
    '''
        Adapted from the official evaluation code
    '''
    truth_dir = os.path.join(path, 'ref')

    if not os.path.exists(truth_dir):
        os.makedirs(truth_dir)

    fact_in_train_annotated = gen_train_facts(os.path.join(path, train_file), truth_dir)
    fact_in_train_distant = gen_train_facts(os.path.join(path, "train_distant.json"), truth_dir)

    truth = json.load(open(os.path.join(path, dev_file)))

    std = {}
    tot_evidences = 0
    titleset = set([])

    title2vectexSet = {}

    for x in truth:
        title = x['title']
        titleset.add(title)

        vertexSet = x['vertexSet']
        title2vectexSet[title] = vertexSet

        if 'labels' not in x:  # official test set from DocRED
            continue

        for label in x['labels']:
            r = label['r']
            h_idx = label['h']
            t_idx = label['t']
            label['evidence'] = []
            std[(title, r, h_idx, t_idx)] = set(label['evidence'])
            tot_evidences += len(label['evidence'])

    tot_relations = len(std)
    tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
    submission_answer = [tmp[0]]

    for i in range(1, len(tmp)):
        x = tmp[i]
        y = tmp[i - 1]
        if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
            submission_answer.append(tmp[i])

    correct_re = 0
    correct_evidence = 0
    pred_evi = 0

    correct_in_train_annotated = 0
    correct_in_train_distant = 0
    titleset2 = set([])
    for x in submission_answer:
        title = x['title']
        h_idx = x['h_idx']
        t_idx = x['t_idx']
        r = x['r']
        titleset2.add(title)
        if title not in title2vectexSet:
            continue
        vertexSet = title2vectexSet[title]

        if 'evidence' in x:  # and (title, h_idx, t_idx) in std:
            evi = set(x['evidence'])
        else:
            evi = set([])
        pred_evi += len(evi)

        if (title, r, h_idx, t_idx) in std:
            correct_re += 1
            stdevi = std[(title, r, h_idx, t_idx)]
            correct_evidence += len(stdevi & evi)
            in_train_annotated = in_train_distant = False
            for n1 in vertexSet[h_idx]:
                for n2 in vertexSet[t_idx]:
                    if (n1['name'], n2['name'], r) in fact_in_train_annotated:
                        in_train_annotated = True
                    if (n1['name'], n2['name'], r) in fact_in_train_distant:
                        in_train_distant = True

            if in_train_annotated:
                correct_in_train_annotated += 1
            if in_train_distant:
                correct_in_train_distant += 1

    re_p = 1.0 * correct_re / len(submission_answer)
    re_r = 1.0 * correct_re / tot_relations if tot_relations != 0 else 0
    if re_p + re_r == 0:
        re_f1 = 0
    else:
        re_f1 = 2.0 * re_p * re_r / (re_p + re_r)

    evi_p = 1.0 * correct_evidence / pred_evi if pred_evi > 0 else 0
    evi_r = 1.0 * correct_evidence / tot_evidences if tot_evidences > 0 else 0

    if evi_p + evi_r == 0:
        evi_f1 = 0
    else:
        evi_f1 = 2.0 * evi_p * evi_r / (evi_p + evi_r)

    re_p_ignore_train_annotated = 1.0 * (correct_re - correct_in_train_annotated) / (
                len(submission_answer) - correct_in_train_annotated + 1e-5)
    re_p_ignore_train = 1.0 * (correct_re - correct_in_train_distant) / (
                len(submission_answer) - correct_in_train_distant + 1e-5)

    if re_p_ignore_train_annotated + re_r == 0:
        re_f1_ignore_train_annotated = 0
    else:
        re_f1_ignore_train_annotated = 2.0 * re_p_ignore_train_annotated * re_r / (re_p_ignore_train_annotated + re_r)

    if re_p_ignore_train + re_r == 0:
        re_f1_ignore_train = 0
    else:
        re_f1_ignore_train = 2.0 * re_p_ignore_train * re_r / (re_p_ignore_train + re_r)

    return [re_p, re_r, re_f1], [evi_p, evi_r, evi_f1], \
        [re_p_ignore_train_annotated, re_r, re_f1_ignore_train_annotated], \
        [re_p_ignore_train, re_r, re_f1_ignore_train]
