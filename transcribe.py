import argparse
import os
import numpy as np
from mido import Message, MidiFile, MidiTrack
from librosa import load
import torch

MIN_MIDI = 21
SAMPLE_RATE = 16000
HOP_LENGTH = SAMPLE_RATE * 20 // 1000


def midi_to_hz(midi):
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def hz_to_midi(freqs):
    return 12.0 * (np.log2(freqs) - np.log2(440.0)) + 69.0

def save_midi(path, pitches, intervals, velocities):
    file = MidiFile()
    track = MidiTrack()
    file.tracks.append(track)
    ticks_per_second = file.ticks_per_beat * 2.0

    events = []
    for i in range(len(pitches)):
        events.append(dict(type='on', pitch=pitches[i], time=intervals[i][0], velocity=velocities[i]))
        events.append(dict(type='off', pitch=pitches[i], time=intervals[i][1], velocity=velocities[i]))
    events.sort(key=lambda row: row['time'])

    last_tick = 0
    for event in events:
        current_tick = int(event['time'] * ticks_per_second)
        velocity = int(event['velocity'] * 127)
        if velocity > 127:
            velocity = 127
        pitch = int(round(hz_to_midi(event['pitch'])))
        track.append(Message('note_' + event['type'], note=pitch, velocity=velocity, time=current_tick - last_tick))
        last_tick = current_tick

    file.save(path)

def get_note_duration(frames):
    bins, T = frames.shape
    assert(bins == 88)
    durs = torch.zeros(frames.shape)
    durs[:,-1] = frames[:,-1]
    for i in range(T-1):
        durs[:, -(i+2)] = (durs[:, -(i+1)] + 1) * frames[:, -(i+2)]

    return durs.to(torch.int)

def extract_notes(onsets, frames, velocity, onset_threshold=0.5, frame_threshold=0.5):
    onsets = onsets.squeeze(0)
    
    velocity = velocity.squeeze(0)
    left = onsets[:1, :] >= onsets[1:2, :]
    right = onsets[-1:, :] >= onsets[-2:-1, :]
    mid = (onsets[1:-1] >= onsets[2:]).float() * (onsets[1:-1] >= onsets[:-2]).float()
    
    
    onsets = torch.cat([left, mid, right], dim=0).float() * onsets
    
    onsets = (onsets > onset_threshold).cpu().to(torch.uint8)
    frames = (frames > frame_threshold).cpu().to(torch.uint8)
    

    onset_diff = torch.cat([onsets[:1, :], onsets[1:, :] - onsets[:-1, :]], dim=0) == 1

    frames = frames.squeeze(0).T

    durs = get_note_duration(frames).T
    
    pitches = []
    intervals = []
    velocities = []

    T = onsets.shape[0]

    for nonzero in onset_diff.nonzero():
        frame = nonzero[0].item()
        pitch = nonzero[1].item()

        onset = frame
        if(onset + 1 >= T):
            offset = onset
        else:
            offset = onset + min(durs[onset+1, pitch], 1000)
        offset = min(offset, T)

        velocity_samples = []
        onset_end = onset
        while onsets[onset_end, pitch].item():
            velocity_samples.append(velocity[onset_end, pitch].item())
            onset_end += 1
            if onset_end >= T:
                break
        pitches.append(pitch)
        intervals.append([onset, max(onset+1, offset)])
        velocities.append(np.mean(velocity_samples) if len(velocity_samples) > 0 else 0)

    return np.array(pitches), np.array(intervals), np.array(velocities)

def load_and_process_audio(flac_path, sequence_length, device):    
    audio, _ = load(flac_path, sr=16000)
    return torch.tensor(audio)

def transcribe(model, audio):
    res = model(audio)
    res['onset'] = torch.squeeze( res['onset'], dim=0 )
    res['frame'] = torch.squeeze( res['frame'], dim=0 )
    return res

def transcribe_file(model_file, flac_paths, save_path, sequence_length,
                  onset_threshold, frame_threshold, device):

    model = torch.load(model_file, map_location=device).eval()
    model.config['device'] = device
    model.inference_mode = True
    
    midi_paths = []
    for flac_path in flac_paths:
        audio = load_and_process_audio(flac_path, sequence_length, device)        
        predictions = transcribe(model, audio)
        predictions['velocity']  = torch.ones_like(predictions['onset'])
        
        p_est, i_est, v_est = extract_notes(predictions['onset'], predictions['frame'], predictions['velocity'], onset_threshold, frame_threshold)
        
        scaling = HOP_LENGTH / SAMPLE_RATE

        i_est = (i_est * scaling).reshape(-1, 2)
        p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_midi(save_path, p_est, i_est, v_est)
        midi_paths.append(save_path)
    return midi_paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--flac_paths', type=str, nargs='+', default="")
    parser.add_argument('--model_file', type=str, default="checkpoints/model-220519-005612-tiny-498000.pt")
    parser.add_argument('--save-path', type=str, default='test')
    parser.add_argument('--sequence-length', default=None, type=int)
    parser.add_argument('--onset-threshold', default=0.4, type=float)
    parser.add_argument('--frame-threshold', default=0.4, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        transcribe_file(**vars(parser.parse_args()))
