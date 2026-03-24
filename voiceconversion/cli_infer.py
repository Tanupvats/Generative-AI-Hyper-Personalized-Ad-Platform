import argparse
import soundfile as sf
from core.engine import RVCEngine
from core.uvr5_separator import VocalSeparator

def main():
    parser = argparse.ArgumentParser(description="Headless RVC Inference CLI")
    parser.add_argument("--input", type=str, required=True, help="Path to input audio")
    parser.add_argument("--output", type=str, required=True, help="Path to save output")
    parser.add_argument("--model", type=str, required=True, help="Path to .pth model")
    parser.add_argument("--index", type=str, default="", help="Path to Faiss .index file")
    parser.add_argument("--pitch_shift", type=int, default=0, help="+12 for M->F, -12 for F->M")
    parser.add_argument("--isolate_vocals", action="store_true", help="Run UVR5 first")
    args = parser.parse_args()

    audio_to_convert = args.input

    if args.isolate_vocals:
        print("Isolating vocals using UVR5...")
        separator = VocalSeparator(model_path="assets/uvr5/HP2_all_vocals.pth")
        vocal_path = "temp_vocals.wav"
        separator.process_audio(args.input, vocal_path, "temp_inst.wav")
        audio_to_convert = vocal_path

    print(f"Loading RVC Engine with model: {args.model}")
    engine = RVCEngine()
    engine.load_hubert("assets/hubert/hubert_base.pt")
    engine.load_vits(args.model)
    if args.index:
        engine.load_faiss_index(args.index)

    print("Converting audio...")
    audio_out, tgt_sr = engine.infer(
        audio_path=audio_to_convert,
        f0_up_key=args.pitch_shift,
        f0_method="rmvpe",
        index_rate=0.75 if args.index else 0.0
    )

    sf.write(args.output, audio_out, tgt_sr)
    print(f"Success! Saved to {args.output}")

if __name__ == "__main__":
    main()