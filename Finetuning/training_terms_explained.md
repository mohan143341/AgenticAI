# Training Terms Made Simple
### epoch · step · batch · gradient_accumulation_steps

A plain-English guide to the four words you keep seeing in finetuning configs.

---

## The one analogy to remember

Imagine you are **studying a book of 1,800 pages** to learn medicine.

| Real life (studying) | Training a model | Term |
|----------------------|------------------|------|
| Read a few pages, then pause to think | Look at a few examples, then learn from them | **batch** |
| One "pause and learn from what you just read" | One weight update | **step** |
| Finishing the whole book once, cover to cover | Going through the whole dataset once | **epoch** |
| Read several small chunks, but only stop to think after a few of them | Collect gradients over a few mini-batches, update once | **gradient_accumulation_steps** |

The model **learns a little bit at every step**, not just at the end of the book (epoch).

---

## 1. Batch

A **batch** is a small group of examples the model looks at *together* before learning.

- We don't feed all 1,800 examples at once — too much memory.
- We feed them in small groups (batches).
- Bigger batch = smoother, more stable learning, but needs more memory.

> **Batch = how many examples the model sees before one round of learning.**

---

## 2. Step

A **step** is **one weight update** — one moment where the model actually changes
and gets a little smarter.

Each step does 4 things:

1. Look at one batch
2. Check how wrong it was (the **loss**)
3. Figure out how to improve (**gradients**)
4. **Update the weights**  ← this is the step

> **Step = one update to the model. Many steps happen inside one epoch.**

---

## 3. Epoch

An **epoch** is **one full pass over the entire dataset** — the model has now
seen every example exactly once.

The link between them:

```
steps per epoch = number of training examples / batch size
```

> **Epoch = "I've now seen all my data one time."**
> Nothing magic happens at the end of an epoch — the learning already
> happened, step by step, along the way.

---

## 4. gradient_accumulation_steps

This is a memory-saving trick.

**The problem:** You want a batch of 8, but your GPU can only hold 1 example
at a time.

**The fix:** Look at 1 example at a time, but *don't update yet*. Remember what
you learned (accumulate the gradients), and only update the weights after 8 of them.

```
effective batch = per_device_batch  ×  gradient_accumulation_steps
                = 1                  ×  8
                = 8
```

So the model **learns as if the batch were 8**, while only ever holding **1**
example in memory.

> **gradient_accumulation_steps = "save up learning from several small groups,
> then update once" — gives you a big batch without needing a big GPU.**

---

## Full worked example (using a real config)

```json
"n_samples": 2000,
"test_size": 0.1,
"per_device_train_batch_size": 1,
"gradient_accumulation_steps": 8,
"num_train_epochs": 1.0,
"max_steps": -1
```

Let's read it step by step:

**Step A — How many examples actually train?**
```
2000 total  ×  (1 − 0.1)  =  1800 training examples
(the other 200 are kept aside for testing)
```

**Step B — What is the effective batch size?**
```
1 (on GPU)  ×  8 (accumulation)  =  8 examples per update
```

**Step C — How many steps in one epoch?**
```
1800 examples  /  8  =  225 steps   (≈ 225 weight updates)
```

**Step D — How many steps total?**
```
225 steps/epoch  ×  1.0 epoch  =  225 steps total
(max_steps = -1 means "don't cap it, let epochs decide")
```

### In one sentence
> The model goes through **1,800 examples once** (1 epoch), updating its
> weights **once every 8 examples**, for a total of about **225 updates**.

---

## Quick reference table

| Term | Simple meaning | In the example |
|------|----------------|----------------|
| **batch** | examples seen before one update | 1 on GPU, 8 effective |
| **gradient_accumulation_steps** | mini-batches saved up before updating | 8 |
| **effective batch** | per_device_batch × accumulation | 8 |
| **step** | one weight update | ~225 of them |
| **steps per epoch** | examples ÷ effective batch | 1800 ÷ 8 = 225 |
| **epoch** | one full pass over the data | 1 |
| **max_steps** | hard cap on total steps (−1 = off) | off |

---

## Two facts people always get wrong

1. **Weights update every STEP, not every epoch.**
   In our example the model improves 225 times, not just once at the end.

2. **Steps per epoch depends on batch size and data size — not on the epoch count.**
   That's why budgeting by `max_steps` is predictable, while "100 epochs"
   can cost wildly different amounts if your data or batch size changes.

---

### The mental formula to memorize

```
effective_batch = per_device_batch × gradient_accumulation_steps × num_gpus

steps_per_epoch = training_examples ÷ effective_batch

total_steps     = steps_per_epoch × num_epochs   (unless max_steps caps it)
```
