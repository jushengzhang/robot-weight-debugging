The key-subspace edit also failed to change task following, so I will reject it. I will test one earlier intervention: increase the contrast of object, destination, spatial-relation, and goal-predicate token embeddings relative to grammatical tokens. This targets instruction content before it mixes with the scene, rather than boosting a late head that may already encode the habitual task.
```python
with notes.open("a") as f:
    f.write(
        "\n## Pass24: L12 semantic-key gain2\n"
        "Dev69->69/128, 0.0 +/-1.9pp, 3 wins/3 losses. "
        "Task5/position3 unchanged, old-goal failures13 unchanged; env17->16, object23->22, "
        "semantic21->23. REJECT. Mean-level semantic key/value subspaces did not repair binding.\n"
    )
for n in list(edited()):
    drop_edit(n)

# Fixed lexical categories: physical referents, spatial qualifiers, goal predicates.
# No episode success labels are used.
content_words = """
black white red yellow blue green bowl plate ramekin basket cabinet drawer drawers
stove microwave refrigerator rack tray shelf top bottom middle upper lower
left right front back behind between next beside above below under inside outside
open close closed turn off on push pull rotate
wooden cream cheese butter milk chocolate pudding tomato sauce bbq barbecue
salad dressing ketchup alphabet soup orange juice bottle wine mug cup pot pan
moka coffee teapot frying book books cookies cookie box package can cans
porcelain ceramic akita brown large small long tall short center corner edge
food container lid handle burners burner
""".split()
function_words = ["the", "a", "an", "and", "it", "is", "of", "to", "please", "then", "that", "this"]
def word_ids(word):
    # Use word-in-context so sentencepiece encodes the word boundary correctly.
    return sp.encode(" " + word, out_type=int)
function_ids = sorted({i for w in function_words for i in word_ids(w)})
content_ids = sorted({i for w in content_words for i in word_ids(w)} - set(function_ids)
                     - {sp.bos_id(), sp.eos_id(), sp.unk_id(), sp.pad_id()})
emb_name = next(n for n in NAMES if n.endswith("language_model.embed_tokens.weight"))
E = load_original(emb_name)
E_function_mean = E[function_ids].mean(0)
selected = E[content_ids].clone()
updated = selected + 0.5*(selected-E_function_mean)
print("Embedding shape:", tuple(E.shape), "changed token rows:", len(content_ids))
print("Content norm median/range:",
      round(float(selected.norm(dim=1).median()),3),
      [round(float(x),3) for x in [selected.norm(dim=1).min(),selected.norm(dim=1).max()]])
print("Function norm median:", round(float(E[function_ids].norm(dim=1).median()),3))
print("Example token IDs/pieces:", [(i,sp.id_to_piece(i).replace(chr(9601),"_").encode("ascii","replace").decode())
                                   for i in content_ids[:25]])
E[content_ids] = updated
save_edit(emb_name, E)
torch.save({"name":emb_name, "ids":content_ids, "original_rows":selected,
            "function_mean":E_function_mean, "gain":1.5}, W/"content_embedding_gain150_recipe.pt")
del E
print("Installed:", edited())
with notes.open("a") as f:
    f.write(
        "\nCandidate K: early lexical-contrast edit. Fixed physical referent/spatial/goal-predicate "
        "vocabulary (no outcome labels). For its "+str(len(content_ids))+" embedding rows, "
        "E_i <- E_i +0.5*(E_i - mean_function_word_embedding). All other rows original. "
        "This changes content direction as well as residual magnitude, so it is not canceled "
        "by the first RMSNorm and is not whole-tensor scaling. "
        "Hypothesis: preserve requested referent signal before scene mixing. "
        "Dose1.5; test task following and already-solved categories.\n"
    )
```
