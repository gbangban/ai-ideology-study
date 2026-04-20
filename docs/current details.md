🛠️ Unsloth: The "Magic" Behind 27B on 32GB
Unsloth is a collection of manually written Triton kernels that replace the standard Hugging Face/PyTorch functions.

    Memory Efficiency: It reduces VRAM usage by up to 70% without losing accuracy. This is why a 27B model (which usually requires 80GB for training) can fit into 32GB.
    Zero-Loss Quantization: Unlike standard bitsandbytes, Unsloth uses 4-bit Dynamic Quantization, which results in 0% degradation in most benchmarks compared to 16-bit.
    Flash Attention 2: It natively integrates Flash Attention 2, which is essential for the RTX 5090’s architecture to maintain speed during long context windows.
    Embedding Fix: In 2024, Unsloth added "Long Context" support, allowing you to train on up to 128k context if you have the VRAM (though for 27B on 32GB, you will stay at 4k).

🏎️ RTX 5090: The Blackwell Advantage
The 5090 is built on the Blackwell architecture, which introduces features you must utilize for your project:

    FP8 Support: The 5090 excels at FP8 (8-bit floating point) training and inference. While you are using 4-bit (QLoRA), you can use FP8 for the Teacher phase to get near-perfect accuracy at incredible speeds.
    VRAM Bandwidth: The move to GDDR7 means your training steps will likely be 30-50% faster than a 4090, even if the "compute" looks similar.
    PCIe 5.0: Ensure your motherboard supports PCIe 5.0; otherwise, the "gradient checkpointing" (which moves data between RAM and VRAM) will become a bottleneck.

🧠 Dialectical Materialism & Alignment (Modern Trends)
Within the last 18 months, "Ideological Alignment" has moved away from simple Instruction Tuning to more robust methods:

    DPO (Direct Preference Optimization): Instead of just showing the model "Correct" DM answers (SFT), you show it a "Correct" answer and a "Liberal/Idealist" answer and tell it which to prefer. This is significantly more effective for ideological grounding.
    Constitutional AI: Modern researchers (like Anthropic) use the Teacher to "critique" its own work based on a Constitution (your DM principles). This prevents the model from just "parroting" terminology without logic.
    System Prompts vs. Fine-tuning: We now know that fine-tuning changes the probability distribution of words (the "voice"), while system prompts control the application of logic. Your plan to do both is the current "Gold Standard."

🔍 Specific Technical "Gotchas" (2024/2025)

    Qwen 2.5/3.5 Architecture: Qwen uses Dense Layers and RoPE (Rotary Positional Embeddings) scaling. When training, you must ensure your training script specifically targets the gate_proj, up_proj, and down_proj layers, or the model's DM logic will be "shallow."
    GGUF Merging: You can now merge LoRA adapters into GGUFs without converting back to FP16 first. This saves you about 40GB of System RAM.
    Linux Kernels: If you are on Ubuntu, you must use Kernel 6.5+ to properly manage the power states and memory address space of a Blackwell-class GPU.