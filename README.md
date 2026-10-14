# AI-Driven Cloud Virtualization & VM Migration Optimizer (QLoRA Fine-Tuning Pipeline)

## Overview

This project implements an end-to-end MLOps and fine-tuning pipeline that transforms a general-purpose Large Language Model (**Mistral-7B-Instruct**) into an autonomous domain expert in **cloud computing and data center virtualization**. 

Instead of generic text generation, the system parses complex, real-world infrastructure logs (Virtual Machines and Physical Machines) and fine-tunes an LLM via **Unsloth 4-bit QLoRA** to predict optimal, constraint-aware VM migrations that minimize energy consumption and balance server workloads.

The workflow covers:
- **Concurrent Data Ingestion:** Multi-process parsing of gigabytes of historical VM JSON files and PM allocation CSV logs.
- **Instruction-Tuned Dataset Generation:** Converting raw cloud operational data into structured system prompts enforcing strict JSON output formats.
- **Parameter-Efficient Fine-Tuning:** Executing 4-bit QLoRA with Unsloth and PEFT to train on a Tesla T4 GPU with minimal memory overhead.
- **Model Merging & Quantization:** Merging LoRA adapters back into the base weights and exporting to GGUF format for edge/local inference via Ollama.

---

## 🎯 Core Motivation & Problem Solved

Cloud data centers face constant trade-offs between maximizing resource utilization (CPU, RAM, Disk) and minimizing power consumption. Traditional heuristic algorithms can be rigid and slow to adapt. 

This project solves this by training a small, specialized LLM to act as an intelligent cloud orchestrator that:
- Respects hard capacity constraints (ensuring every VM is successfully allocated without overloading physical hosts).
- Balances CPU and memory bottlenecks dynamically.
- Minimizes overall physical machine energy usage.
- Outputs clean, code-ready JSON dictionaries mapping virtual machines directly to optimal physical nodes.

---

## 🛠️ Tech Stack

- **Python** (Pandas, NumPy, Concurrent.futures)
- **Machine Learning / LLM:** PyTorch, Hugging Face Transformers, TRL, Unsloth, PEFT
- **Quantization & Deployment:** GGUF (`q4_k_m`), Ollama
- **Environment:** Google Colab (GPU T4)

---

## ⚙️ Pipeline Architecture

1. **Data Preprocessing & Multi-Process Parsing:** Uses `ProcessPoolExecutor` to parse distributed JSON configurations (`x_virtual_machines.json`, `x_physical_machines.json`) and cleaned CSV allocation outputs (`y_xxxx.csv`).
2. **Dynamic Prompt Construction:** Formulates expert system prompts encapsulating CPU, RAM, disk storage, and energy constraints alongside JSON data payloads.
3. **4-Bit QLoRA Fine-Tuning:** Leverages Unsloth's optimized patching to train target attention and MLP modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) with reduced VRAM usage.
4. **Adapter Merging:** Seamlessly merges the trained PEFT LoRA weights back into the 16-bit/4-bit base model structure.
5. **GGUF Export & Ollama Conversion:** Quantizes the merged model into `unsloth.Q4_K_M.gguf` for high-performance local deployment.

---

## 📊 Key Features

- **High Training Efficiency:** Achieved rapid convergence over 40 training steps with a smooth loss drop down to ~0.08 using `adamw_8bit`.
- **Strict Output Grounding:** Tuned instruction templates to eliminate conversational filler, guaranteeing strict valid Python dictionary/JSON outputs for programmatic consumption.
- **Production-Ready Artifacts:** Complete pipeline script supporting local checkpointing, Google Drive persistence, and Ollama integration.

---

## 🚀 How to Run

1. Open the notebook in Google Colab (with GPU accelerator enabled).
2. Mount your Google Drive containing the structured dataset directories.
3. Run the cells sequentially to parse logs, construct the dataset, execute the Unsloth trainer, and export the GGUF model.

---

## Project Structure

```text
notebook/
    finetuning_pipeline.ipynb   # Complete data loading, QLoRA training, merging, and GGUF export script
