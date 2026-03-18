import os
import sys

# Ensure local modules are discoverable
sys.path.append(os.path.abspath("src"))
sys.path.append(os.path.abspath("benchmarks"))

try:
    from tracking_efficiency import run_benchmark as run_tracking
    from compression_performance import run_compression_benchmark as run_compression
except ImportError:
    # Fallback if the path logic above is tricky in some environments
    print("Warning: Could not import benchmarks. Ensure you are in the project root.")
    sys.exit(1)

def get_videos(data_dir="data"):
    """
    Scans the data directory for common video formats.
    """
    valid_exts = ('.mp4', '.avi', '.mkv', '.mov', '.y4m')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        return []
    return sorted([f for f in os.listdir(data_dir) if f.lower().endswith(valid_exts)])

def main():
    videos = get_videos()
    
    if not videos:
        print("\n[!] No video files found in 'data/' folder.")
        print("    Please place some video files (.mp4, .avi, etc.) in the 'data/' directory.")
        print("    Example: data/foreman.mp4")
        return

    print("\n" + "═"*40)
    print("  COUTINHO 2017 - VIDEO TEST SELECTOR")
    print("═"*40)
    
    print("  Available Videos:")
    for i, v in enumerate(videos, 1):
        print(f"  {i:2}. {v}")
    
    all_index = len(videos) + 1
    print(f"  {all_index:2}. [RUN ALL VIDEOS]")
    print("   0. Exit")
    print("─"*40)

    try:
        video_choice = input("\nSelect video number(s) to test (e.g. '1', '1, 3' or 'all'): ").strip().lower()
        
        if video_choice == '0':
            print("Exiting...")
            return
            
        selected_videos = []
        if video_choice in ['all', 'a', str(all_index)]:
            selected_videos = videos
        else:
            indices = [int(x.strip()) for x in video_choice.replace(',', ' ').split()]
            for idx in indices:
                if 1 <= idx <= len(videos):
                    selected_videos.append(videos[idx-1])
                else:
                    print(f"  [!] Warning: Index {idx} is out of range.")

        if not selected_videos:
            print("  [!] No valid videos selected.")
            return

        print("\n  Available Benchmarks:")
        print("  1. Tracking Efficiency (PBM Metric)")
        print("  2. Compression Performance (PSNR/SSIM)")
        bench_choice = input("\nSelect benchmark (1 or 2): ").strip()

        print(f"\n  Starting processing for {len(selected_videos)} video(s)...")

        for video in selected_videos:
            video_path = os.path.join("data", video)
            print("\n" + "█"*60)
            print(f"  TESTING: {video}")
            print("█"*60)
            
            try:
                if bench_choice == '2':
                    run_compression(video_path=video_path)
                else:
                    run_tracking(video_path=video_path)
            except Exception as e:
                print(f"  [!] Error processing {video}: {e}")

        print("\n" + "═"*40)
        print("  BATCH PROCESSING COMPLETE")
        print("═"*40)

    except ValueError:
        print("\n  [!] Invalid input. Please enter numbers, 'all', or '0'.")
    except KeyboardInterrupt:
        print("\n\n  Interrupted by user. Exiting.")

if __name__ == "__main__":
    main()
