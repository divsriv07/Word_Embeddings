"""
wordsim.py
----------

Evaluates word embeddings using the WordSim-353 dataset:

1. Skips word pairs where either of the two words is not in the vocabulary.
2. Computes cosine similarity only for valid pairs.
3. Calculates Spearman’s Rank Correlation on the valid pairs.
4. Reports the mean and variance of predicted similarities over valid pairs.
5. Saves the valid pairs, human scores, and predicted similarities to a CSV file.

Usage:
    python wordsim.py --embedding svd.pt
    python wordsim.py --embedding cbow.pt
    python wordsim.py --embedding skipgram.pt
"""

import argparse
import torch
import statistics
from scipy.stats import spearmanr

def loadEmbedding(embeddingPath):
    """
    Loads embeddings and vocabulary mappings from the specified file.

    Args:
        embeddingPath (str): Path to the saved embedding file.

    Returns:
        embeddings (Tensor): Embedding matrix or vector set.
        wordToIdx (dict): Mapping of word to index.
        idxToWord (dict): Mapping of index to word.
    """
    checkpoint = torch.load(embeddingPath)
    
    # If "embeddings" key exists, it's likely the SVD model
    if 'embeddings' in checkpoint:
        return checkpoint['embeddings'], checkpoint['wordToIdx'], checkpoint['idxToWord']
    else:
        # For CBOW/Skip-Gram, we use the input embedding
        return checkpoint['inEmbedding'], checkpoint['wordToIdx'], checkpoint['idxToWord']

def cosineSimilarity(vec1, vec2):
    """
    Computes cosine similarity between two vectors.

    Args:
        vec1 (Tensor): First vector.
        vec2 (Tensor): Second vector.

    Returns:
        float: Cosine similarity value in [-1, 1].
    """
    numerator = torch.dot(vec1, vec2).item()
    denominator = (vec1.norm(2) * vec2.norm(2)).item()
    if denominator == 0:
        return 0.0
    return numerator / denominator

def evaluateWordSim(embeddingPath, wordSimFile='wordsim353.csv'):
    """
    Evaluates embeddings using the WordSim-353 dataset by:
      1. Skipping word pairs where either word is missing from the vocab.
      2. Computing cosine similarity only for valid pairs.
      3. Calculating Spearman’s Rank Correlation on these valid pairs.
      4. Computing mean and variance of predicted similarities.
      5. Saving valid results to a CSV file.

    Args:
        embeddingPath (str): Path to the embedding file (e.g., 'svd.pt', 'cbow.pt', etc.).
        wordSimFile (str): CSV file with columns: word1,word2,score
    """
    # Load embeddings
    embeddings, wordToIdx, idxToWord = loadEmbedding(embeddingPath)
    
    # Ensure embeddings on CPU
    if embeddings.is_cuda:
        embeddings = embeddings.cpu()

    # Load the WordSim-353 data
    allWordPairs = []
    allGoldScores = []
    with open(wordSimFile, 'r', encoding='utf-8') as f:
        # Skip header if present
        next(f, None)
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 3:
                continue
            w1, w2, score = parts[0].lower(), parts[1].lower(), float(parts[2])
            allWordPairs.append((w1, w2))
            allGoldScores.append(score)

    # Filter pairs and compute similarities
    filteredWordPairs = []
    filteredGoldScores = []
    predScores = []

    for (w1, w2), gold in zip(allWordPairs, allGoldScores):
        if w1 not in wordToIdx or w2 not in wordToIdx:
            # Skip if either word not in vocab
            continue
        
        vec1 = embeddings[wordToIdx[w1]]
        vec2 = embeddings[wordToIdx[w2]]
        sim = cosineSimilarity(vec1, vec2)
        
        filteredWordPairs.append((w1, w2))
        filteredGoldScores.append(gold)
        predScores.append(sim)

    if len(filteredGoldScores) == 0:
        print(f"No valid word pairs found in {wordSimFile} with the current vocabulary!")
        return

    # Calculate Spearman's Rank Correlation
    correlation, _ = spearmanr(filteredGoldScores, predScores)

    # Compute mean and variance of the predicted similarities
    meanSim = statistics.mean(predScores)
    # For variance, set ddof=1 (sample variance) if desired
    varSim = statistics.pvariance(predScores)  # population variance

    # Print summary
    print("=============================================")
    print(f"Embedding: {embeddingPath}")
    print(f"Total pairs in dataset: {len(allWordPairs)}")
    print(f"Valid pairs used: {len(filteredWordPairs)}")
    print(f"Spearman’s Rank Correlation: {correlation:.4f}")
    print(f"Mean of predicted similarities: {meanSim:.4f}")
    print(f"Variance of predicted similarities: {varSim:.4f}")
    print("=============================================")

    # Save valid results to CSV (with mean/variance appended at the end)
    outputFile = f"{embeddingPath}_wordsim_results.csv"
    with open(outputFile, "w", encoding="utf-8") as fw:
        fw.write("word1,word2,gold,similarity\n")
        for (w1, w2), gold, sim in zip(filteredWordPairs, filteredGoldScores, predScores):
            fw.write(f"{w1},{w2},{gold},{sim}\n")

        # Optionally, write mean and variance at the bottom 
        fw.write(f"\n# SpearmanCorrelation,{correlation}\n")
        fw.write(f"# MeanPredSim,{meanSim}\n")
        fw.write(f"# VarPredSim,{varSim}\n")

    print(f"Results saved to '{outputFile}'.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding', type=str, required=True,
                        help='Path to the embedding file (svd.pt, cbow.pt, skipgram.pt).')
    parser.add_argument('--wordsim-file', type=str, default='wordsim353.csv',
                        help='Path to the WordSim-353 CSV file.')
    args = parser.parse_args()

    evaluateWordSim(args.embedding, args.wordsim_file)

if __name__ == "__main__":
    main()
