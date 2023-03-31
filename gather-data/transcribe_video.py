import argparse
import os
import subprocess
import csv
from pyannote.audio import Audio
from pyannote.core import Segment
import whisper
from pyannote.audio import Pipeline


def download_video(url, cookies_path, video_dir):
    cmd = f"yt-dlp -f bestaudio --cookies {cookies_path} -o {video_dir}/video.webm {url}"
    subprocess.run(cmd, shell=True, check=True)


def convert_webm_to_wav(video_dir):
    cmd = f"ffmpeg -i {video_dir}/video.webm -ac 1 -ar 16000 {video_dir}/video.wav"
    subprocess.run(cmd, shell=True, check=True)


def split_wav(video_dir):
    cmd = f"ffmpeg -i {video_dir}/video.wav -f segment -segment_time 3600 -c copy {video_dir}/chunk_%03d.wav"
    subprocess.run(cmd, shell=True, check=True)


def transcribe_chunk(filename, video_dir, transcription_dir, model, min_speaker, max_speaker, num_speaker):
    input_file = os.path.join(video_dir, filename)
    speaker_diarization = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                                   use_auth_token=True)
    who_speaks_when = speaker_diarization(
        input_file, num_speakers=num_speaker, min_speakers=min_speaker, max_speakers=max_speaker)

    who_speaks_when = model.who_speaks_when(input_file, num_speaker)
    audio = Audio(sample_rate=16000, mono=True)

    with open(os.path.join(transcription_dir, f"{filename}_transcription.csv"), "w", newline="") as csvfile:
        fieldnames = ["start", "end", "speaker", "text"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for segment, _, speaker in who_speaks_when.itertracks(yield_label=True):
            waveform, _ = audio.crop(input_file, segment)
            text = model.transcribe(waveform.squeeze().numpy())["text"]
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
        os.makedirs(args.video_dir)

    if not os.path.exists(args.transcription_dir):
        os.makedirs(args.transcription_dir)

    model = whisper.load_model(args.model)

    download_video(args.url, args.cookies, args.video_dir)
    convert_webm_to_wav(args.video_dir)
    split_wav(args.video_dir)

    for chunk_filename in sorted(os.listdir(args.video_dir)):
        if chunk_filename.startswith("chunk_") and chunk_filename.endswith(".wav"):
            transcribe_chunk(chunk_filename, args.video_dir, args.transcription_dir,
                             model, args.min_speaker, args.max_speaker, args.num_speaker)


if __name__ == "__main__":
    main()
