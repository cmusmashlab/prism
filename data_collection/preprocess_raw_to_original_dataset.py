"""
Preprocess the raw dataset to be used in the task.

Before:
- datadrive / tasks / {task_name} / dataset / raw (cannot be publicly shared)

After:
- datadrive / tasks / {task_name} / dataset / original (can be publicly shared)
"""

import argparse
import librosa
import os
import pandas as pd
import soundfile
import shutil


from prism import config
from prism.har import params

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, help='Task name', required=True)
    parser.add_argument('--sid', type=str, help='Session ID. If None, process all sessions.', default=None)
    parser.add_argument('--video-export', action='store_true', help='If True, it exports video.')
    parser.add_argument('--merge-label',  action='store_true', help='If True, it merges the labels as defined in the file provided at dataset/merge.csv')
    return parser.parse_args()


def process_one_session(session_id):
    print('Processing: ', session_id)
    raw_dir = dataset_dir / 'raw' / session_id
    original_dir = dataset_dir / 'original' / session_id
    original_dir.mkdir(parents=True, exist_ok=True)

    # annotation
    if args.task in ['latte_making', 'cooking']:
        # copy the annotation file from the previous original directory
        prev_annotation_fp = dataset_dir / 'prev_original' / session_id / 'annotation.txt'
        if prev_annotation_fp.exists():
            shutil.copy(prev_annotation_fp, original_dir / 'annotation.txt')
        else:
            print(f'Annotation file {prev_annotation_fp} does not exist. Skipping this session.')
            os.system(f'rm -rf {original_dir}')
            return
    elif args.task in ['MakeCoffee', 'MakeTea', 'MakeCereal', 'MakeSandwich', 'MakeStencil', 'skin_care', 'uist_demo']:
        annotation_files = list(raw_dir.glob('*annotation*.csv'))
        if len(annotation_files) == 1:
            success, annotation, video_begin_time, video_end_time = parse_annotation_from_via(annotation_files[0])
            if not success:
                print('Error in parsing annotation. Skipping this session.')
                os.system(f'rm -rf {original_dir}')
                return
            annotation.to_csv(original_dir / 'annotation.txt', index=False)

            # video
            if args.video_export:
                video_path = raw_dir / 'video.mp4'
                assert video_path.exists(), f'Video file {video_path} does not exist.'
                video_output_path = original_dir / 'video.mp4'
                start_time = video_begin_time / 1000
                end_time = video_end_time / 1000
                os.system(f'ffmpeg -ss {start_time} -i {video_path} -to {end_time} -c:v copy -c:a copy {video_output_path}')
        else:
            print(f'Found {len(annotation_files)} annotation files in {raw_dir}. Expected exactly 1. Skipping this session.')
            os.system(f'rm -rf {original_dir}')
            return

    # get clap time for audio + motion trimming
    clap_time_fp = raw_dir / 'clap_time_audio.txt'
    if clap_time_fp.exists():
        with open(raw_dir / 'clap_time_audio.txt', 'r') as f:
            line = f.readline().strip()
            if ',' in line:
                clap_time_start_ms = float(line.split(',')[0])
                clap_time_end_ms = float(line.split(',')[1])
            else:
                clap_time_start_ms = float(line)
                clap_time_end_ms = None
        print('Trimming audio + motion from', clap_time_start_ms, 'to', clap_time_end_ms)
    else:
        clap_time_start_ms = 0
        clap_time_end_ms = None
        print('No audio trimming because no clap_time_audio.txt is found')

    # audio
    # TODO: We want to check different audio file format (librosa vs wavfile)
    audio_files = list(raw_dir.glob('*.wav'))
    assert len(audio_files) == 1, f'Found {len(audio_files)} audio files in {raw_dir}. Expected at most 1.'

    audio, _ = librosa.load(audio_files[0], sr=params.SAMPLE_RATE, mono=True)  # force resampling 
    if clap_time_end_ms is not None:
        audio = audio[int(clap_time_start_ms * params.SAMPLE_RATE / 1000): int(clap_time_end_ms * params.SAMPLE_RATE / 1000)]
    else:
        audio = audio[int(clap_time_start_ms * params.SAMPLE_RATE / 1000):]
    soundfile.write(original_dir / 'audio.wav', audio, samplerate=params.SAMPLE_RATE)

    # motion
    sensor_columns = ['timestamp', 'userAcceleration.x', 'userAcceleration.y', 'userAcceleration.z',
                    'gravity.x', 'gravity.y', 'gravity.z',
                    'rotationRate.x', 'rotationRate.y', 'rotationRate.z',
                    'magneticField.field.x', 'magneticField.field.y', 'magneticField.field.z',
                    'attitude.roll', 'attitude.pitch', 'attitude.yaw',
                    'attitude.quaternion.x', 'attitude.quaternion.y', 'attitude.quaternion.z',
                    'attitude.quaternion.w', 'sensor_time']
    motion_files = list(raw_dir.glob('*motion*.txt'))
    assert len(motion_files) == 1, f'Found {len(motion_files)} motion files in {raw_dir}. Expected at most 1.'
    motion = pd.read_csv(motion_files[0], sep='\s', engine='python', index_col=False)
    if motion.shape[1] - len(sensor_columns) == 1:  # this is an old version of the motion data (nan is included)
        motion = motion.dropna(axis=1)
    assert motion.shape[1] == len(sensor_columns)
    motion.columns = sensor_columns
    motion = motion.drop('sensor_time', axis=1)

    ## crop chronologically
    motion['timestamp'] *= 1000
    motion['timestamp'] -= motion['timestamp'].to_list()[0]
    motion['timestamp'] -= clap_time_start_ms
    if clap_time_end_ms is not None:
        motion = motion[(motion['timestamp'] >= 0) & (motion['timestamp'] <= clap_time_end_ms - clap_time_start_ms)]
    else:
        motion = motion[motion['timestamp'] >= 0]
    motion.to_csv(original_dir / 'motion.txt', index=False, sep=' ')
    print('Processed: ', session_id)


def parse_annotation_from_via(annotation_fp):
    """
    Parse the annotation from the VIA CSV file. This is the standard way in most of the tasks.
    Args:
    * annotation_fp (str): path to the VIA CSV file.

    Returns:
    * success (bool): whether the parsing is successful.
    * annotation (pd.DataFrame): a DataFrame containing the parsed annotation.
    * clap_time (float): the time of the clap in video in milliseconds.
    * end_time (float): the time of the end in video in milliseconds.
    """
    # read the annotation file
    try:
        annotation = pd.read_csv(annotation_fp, comment='#', header=None)
        annotation.columns = ['metadata_id', 'file_list', 'flags', 'temporal_coordinates', 'spatial_coordinates', 'metadata']
        if annotation['metadata_id'][0] == 'metadata_id':
            annotation = annotation.drop(0)  # drop the first row if it is a header
    except Exception as e:
        print('Error in reading annotation csv:', e)
        print('Skipping the process.')
        return False, None, None, None
    
    # check if the merge file exists
    if args.merge_label:
        merge_file = dataset_dir / 'merge.csv'
        try:
            merge_df = pd.read_csv(merge_file, header=None)
            merge_df.columns = ['from', 'to']      
            print(f'Merging labels using {merge_file}')     
        except Exception as e:
            print(f'Error in reading merge file {merge_file}: {e}')
            return False, None, None, None

    tmp_annotation_dict = {}
    for _, row in annotation.iterrows():
        try:
            start_time_sec = float(row['temporal_coordinates'].split(',')[0][1:])
            label = row['metadata'].split('\"')[3]
            if label != 'END':
                label = label.lower()
            if args.merge_label:
                if label in merge_df['from'].values:
                    label = merge_df[merge_df['from'] == label]['to'].values[0]
            tmp_annotation_dict[label] = start_time_sec * 1000
        except Exception as e:
            print('Error in parsing annotation:', e)
            print('Row:', row)
            print('Skipping this row.')
            continue
    annotation_dict = {'Timestamp': [], 'Step': []}
    for i, (label, time) in enumerate(sorted(tmp_annotation_dict.items(), key=lambda x: x[1])):
        if i == 0:
            assert label == 'clap', 'The first label is wrong: ' + label
            annotation_dict['Timestamp'].append(0)
            annotation_dict['Step'].append('BEGIN')
        else:
            annotation_dict['Timestamp'].append(time - tmp_annotation_dict['clap'])
            annotation_dict['Step'].append(label)
    return True, pd.DataFrame(annotation_dict), tmp_annotation_dict['clap'], tmp_annotation_dict['END']

if __name__ == '__main__':
    args = get_args()
    dataset_dir = config.datadrive / 'tasks' / args.task / 'dataset'
    if args.sid is not None:
        process_one_session(args.sid)
    else:
        for session_id in os.listdir(dataset_dir / 'raw'):
            if '.DS_Store' in session_id:
                continue
            if not (dataset_dir / 'original' / session_id).exists():
                process_one_session(session_id)
            else:
                print(dataset_dir / 'original' / session_id)
                print('Already processed:', session_id)