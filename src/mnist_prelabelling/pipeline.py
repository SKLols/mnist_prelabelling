"""
Main pipeline orchestrator for MNIST pre-labelling.

Usage:
    python src/mnist_prelabelling/pipeline.py
    # or after pip install -e .:
    mnist-prelabelling
"""

import torch
from mnist_prelabelling.config import get_run_output_dir
from mnist_prelabelling.training.generate_pool_embeddings import run as generate_embeddings
from mnist_prelabelling.clustering.cluster_embeddings import run as cluster
from mnist_prelabelling.clustering.visualize_embeddings import run as visualize
from mnist_prelabelling.clustering.assign_labels import run as assign_labels
from mnist_prelabelling.training.retrain_on_pool_labels import run as retrain
from mnist_prelabelling.clustering.detect_label_errors import run as detect_errors
from mnist_prelabelling.clustering.correct_labels import run as correct
from mnist_prelabelling.evaluation.evaluate_prelabelling import run as evaluate


def main():
    print("\n" + "="*60)
    print("MNIST PRE-LABELLING PIPELINE")
    print("="*60 + "\n")

    # Create one timestamped output directory for this entire run
    output_dir = str(get_run_output_dir())
    print(f"Run outputs will be saved to: {output_dir}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    print("--- Generating pool embeddings ---")
    cnn_model, pool_embeddings, device = generate_embeddings(
        device=device, output_dir=output_dir
    )

    print("\n--- Clustering embeddings ---")
    cluster_ids, distances_to_centroid, centroids = cluster(
        pool_embeddings=pool_embeddings, output_dir=output_dir
    )

    print("\n--- Visualizing embeddings ---")
    visualize(
        pool_embeddings=pool_embeddings,
        cluster_ids=cluster_ids,
        output_dir=output_dir
    )

    print("\n--- Assigning labels via Hungarian algorithm ---")
    pool_initial_labels, cluster_to_digit = assign_labels(
        cnn_model=cnn_model,
        pool_embeddings=pool_embeddings,
        cluster_ids=cluster_ids,
        centroids=centroids,
        device=device,
        output_dir=output_dir,
    )

    print("\n--- Retraining on self-generated labels ---")
    confidence_per_epoch, device = retrain(
        pool_initial_labels=pool_initial_labels,
        device=device,
        output_dir=output_dir,
    )

    print("\n--- Detecting label errors ---")
    flag_confidence, flag_distance = detect_errors(
        confidence_per_epoch=confidence_per_epoch,
        distances_to_centroid=distances_to_centroid,
        cluster_ids=cluster_ids,
        output_dir=output_dir,
    )

    print("\n--- Correcting label errors (iterative) ---")
    pool_corrected_labels = correct(
        pool_embeddings=pool_embeddings,
        pool_initial_labels=pool_initial_labels,
        flag_confidence=flag_confidence,
        flag_distance=flag_distance,
        distances_to_centroid=distances_to_centroid,
        cluster_ids=cluster_ids,
        device=device,
        output_dir=output_dir,
    )
    
    print("\n--- Final evaluation ---")
    accuracy_before, accuracy_after = evaluate(
        pool_initial_labels=pool_initial_labels,
        pool_corrected_labels=pool_corrected_labels,
        output_dir=output_dir,
    )

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print(f"Outputs saved to: {output_dir}")
    print(f"Pre-labelling accuracy: {accuracy_before*100:.2f}% → {accuracy_after*100:.2f}%")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()