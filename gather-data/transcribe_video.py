import argparse
import os
import subprocess
import csv
from pyannote.audio import Audio
from pyannote.core import Segment
import whisper
from pyannote.audio import Pipeline


def extract_video_id(url):
    video_id = url.split("=")[-1]
    return video_id


def download_video(url, cookies_path, video_dir, video_id):
    cmd = f"yt-dlp -f bestaudio --cookies {cookies_path} -o {video_dir}/{video_id}_video.webm {url}"
    subprocess.run(cmd, shell=True, check=True)


def convert_webm_to_wav(video_dir, video_id):
    cmd = f"ffmpeg -y -i {video_dir}/{video_id}_video.webm -ac 1 -ar 16000 {video_dir}/{video_id}_video.wav"
    subprocess.run(cmd, shell=True, check=True)


def split_wav(video_dir, video_id):
    cmd = f"ffmpeg -y -i {video_dir}/{video_id}_video.wav -f segment -segment_time 3600 -c copy {video_dir}/{video_id}_chunk_%03d.wav"
    subprocess.run(cmd, shell=True, check=True)


def transcribe_chunk(filename, video_dir, transcription_dir, model, min_speaker, max_speaker, num_speaker):
    input_file = os.path.join(video_dir, filename)
    print(f"     Loading speaker diarization model...\n")
    speaker_diarization = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                                   use_auth_token=True)
    who_speaks_when = speaker_diarization(
        input_file, num_speakers=num_speaker, min_speakers=min_speaker, max_speakers=max_speaker)

    audio = Audio(sample_rate=16000, mono=True)

    with open(os.path.join(transcription_dir, f"{filename}_transcription.csv"), "w", newline="", encoding='utf-8') as csvfile:
        fieldnames = ["start", "end", "speaker", "text"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        print(f"     Start transcribing {filename}...\n")
        for segment, _, speaker in who_speaks_when.itertracks(yield_label=True):
            waveform, _ = audio.crop(input_file, segment)
            text = model.transcribe(waveform.squeeze().numpy())["text"]
            print(
                f"    {segment.start:06.1f}s - {segment.end:06.1f}s {speaker}: {text}")

            writer.writerow({"start": f"{segment.start:06.1f}s",
                            "end": f"{segment.end:06.1f}s", "speaker": speaker, "text": text})


def main():
    parser = argparse.ArgumentParser(
        description="Download and transcribe a YouTube video.")
    parser.add_argument("url", help="Video URL")
    parser.add_argument("--cookies", default="./cookies.txt",
                        help="Path to cookies file")
    parser.add_argument("--video_dir", default="./videos",
                        help="Output directory for videos")
    parser.add_argument("--transcription_dir", default="./transcriptions",
                        help="Output directory for transcriptions")
    parser.add_argument("--model", default="small", help="Whisper ASR model")
    parser.add_argument("--min_speaker", default=2, type=int,
                        help="Minimum number of speakers")
    parser.add_argument("--max_speaker", default=2, type=int,
                        help="Maximum number of speakers")
    parser.add_argument("--num_speaker", default=2,
                        type=int, help="Number of speakers")

    args = parser.parse_args()

    if not os.path.exists(args.video_dir):
        print(f"    Creating directory {args.video_dir}...")
        os.makedirs(args.video_dir)

    if not os.path.exists(args.transcription_dir):
        print(f"    Creating directory {args.transcription_dir}...")
        os.makedirs(args.transcription_dir)

    print("    Loading whisper...")
    model = whisper.load_model(args.model)

    video_id = extract_video_id(args.url)
    print(f"    Downloading video {video_id}...")
    download_video(args.url, args.cookies, args.video_dir, video_id)

    print(f"    Converting video {video_id} to WAV...")
    convert_webm_to_wav(args.video_dir, video_id)

    print(f"    Splitting video {video_id} into 1 hour chunks...")
    split_wav(args.video_dir, video_id)

    for chunk_filename in sorted(os.listdir(args.video_dir)):
        if chunk_filename.startswith(f"{video_id}_chunk_") and chunk_filename.endswith(".wav"):
            print(f"    Transcribing chunk {chunk_filename}...")
            transcribe_chunk(chunk_filename, args.video_dir, args.transcription_dir,
                             model, args.min_speaker, args.max_speaker, args.num_speaker)


if __name__ == "__main__":
    main()
