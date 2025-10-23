"""Debug COGS encoding to understand why retrieval fails."""

from model_structural_hrr import StructuralHRRModel
from test_cogs_structural import load_cogs


def debug_encoding():
    print("="*70)
    print("DEBUGGING COGS ENCODING")
    print("="*70)
    print()

    # Load small dataset
    train_data = load_cogs('train', limit=10)

    model = StructuralHRRModel(hrr_dim=2048)

    # Store training examples
    for sentence_tokens, output_str in train_data:
        structure = model.parse_cogs_output(output_str)
        model.store(sentence_tokens, structure)

    print(f"Stored {len(model.examples)} examples")
    print()

    # Test on the SAME examples to see if we can retrieve them
    print("Testing retrieval on TRAINING examples (should be perfect):")
    print("-"*70)

    correct = 0
    for i, (sent_tokens, expected_str) in enumerate(train_data):
        expected_struct = model.parse_cogs_output(expected_str)
        predicted_struct = model.retrieve(sent_tokens)

        match = (str(predicted_struct) == str(expected_struct))
        if match:
            correct += 1

        status = "[OK]" if match else "[FAIL]"
        sent_str = ' '.join(sent_tokens[:6])
        print(f"{i+1:2}. {status} {sent_str[:50]}")

        if not match:
            # Show similarity scores
            query_vec = model.sentence_encodings[i]
            print(f"     Similarity to self: {model.vocab.hrr.similarity(query_vec, query_vec):.4f}")

            # Find top 3 matches
            sims = []
            for j, sent_vec in enumerate(model.sentence_encodings):
                sim = model.vocab.hrr.similarity(query_vec, sent_vec)
                sims.append((j, sim))

            sims.sort(key=lambda x: x[1], reverse=True)
            print(f"     Top matches: {[(idx, f'{sim:.4f}') for idx, sim in sims[:3]]}")

    print()
    print(f"Training set retrieval accuracy: {correct}/{len(train_data)}")
    print()

    # Test sentence encoding: are different sentences actually different?
    print("Checking sentence encoding diversity:")
    print("-"*70)
    all_sims = []
    for i in range(len(model.sentence_encodings)):
        for j in range(i+1, len(model.sentence_encodings)):
            sim = model.vocab.hrr.similarity(
                model.sentence_encodings[i],
                model.sentence_encodings[j]
            )
            all_sims.append(sim)

    if all_sims:
        avg_sim = sum(all_sims) / len(all_sims)
        max_sim = max(all_sims)
        min_sim = min(all_sims)
        print(f"Pairwise similarity: avg={avg_sim:.4f}, min={min_sim:.4f}, max={max_sim:.4f}")
        print(f"Total comparisons: {len(all_sims)}")

    print()
    print("="*70)


if __name__ == '__main__':
    debug_encoding()
