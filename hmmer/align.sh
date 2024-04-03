#!/bin/bash

# Prompt user for input directory containing FASTA files
read -p "Enter directory containing FASTA files: " fasta_dir

# Prompt user for custom Phmmer database path
read -p "Enter path to your custom Phmmer database: " phmmer_db


output_dir="${fasta_dir}_output"

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Loop through each FASTA file in the directory
for fasta_file in "$fasta_dir"/*.fasta; do
    # Extract the base name of the file for naming the output
    base_name=$(basename "$fasta_file" .fasta)

    # Run Phmmer for each sequence
    phmmer -o "$output_dir/${base_name}.txt" "$fasta_file" "$phmmer_db"

    # Optional: Echo a message to indicate progress
    echo "Phmmer done for $base_name"
done

echo "All Phmmer searches are complete!"
