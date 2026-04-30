from sentence_transformers import SentenceTransformer, util
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_dynamic_threshold(cosine_matrix):
    if cosine_matrix.size == 0:
        return 0.50 

    max_scores = np.max(cosine_matrix, axis=1)
    mean_score = np.mean(max_scores)
    std_score = np.std(max_scores)
    
    threshold = max(0.50, mean_score - (0.5 * std_score))
    return threshold

def evaluate_extraction(extracted, ground_truth):
    if not extracted or not ground_truth:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "threshold": 0.0}

    ext_strs = [f"{t['subject']} {t['relation']} {t['obj']}" for t in extracted]
    gt_strs = [f"{t['subject']} {t['relation']} {t['obj']}" for t in ground_truth]

    ext_embs = model.encode(ext_strs, convert_to_tensor=True)
    gt_embs = model.encode(gt_strs, convert_to_tensor=True)

    cosine_scores = util.cos_sim(ext_embs, gt_embs).cpu().numpy()
    
    threshold = get_dynamic_threshold(cosine_scores)

    scores_flat = []

    for i in range(len(extracted)):
        for j in range(len(ground_truth)):
            scores_flat.append((cosine_scores[i][j], i, j))
            
    scores_flat.sort(key=lambda x: x[0], reverse=True)

    matched_ext = set()
    matched_gt = set()
    true_positives = 0

    for score, i, j in scores_flat:
        if score >= threshold and i not in matched_ext and j not in matched_gt:
            true_positives += 1
            matched_ext.add(i)
            matched_gt.add(j)

    precision = true_positives / len(extracted) if len(extracted) > 0 else 0
    recall = true_positives / len(ground_truth) if len(ground_truth) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "threshold_used": round(float(threshold), 3)
    }