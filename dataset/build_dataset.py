import json

import pandas as pd

gt_labels = ['Background', 'Church bell', 'Male speech, man speaking', 'Bark', 'Fixed-wing aircraft, airplane',
             'Race car, auto racing', 'Female speech, woman speaking', 'Helicopter', 'Violin, fiddle', 'Flute',
             'Ukulele', 'Frying (food)', 'Truck', 'Shofar', 'Motorcycle', 'Acoustic guitar', 'Train horn',
             'Clock', 'Banjo', 'Goat', 'Baby cry, infant cry', 'Bus', 'Chainsaw', 'Cat', 'Horse', 'Toilet flush',
             'Rodents, rats, mice', 'Accordion', 'Mandolin']


def csv_to_json(csv_path, json_path):
    csv = pd.read_csv(csv_path, sep='&')
    label = csv.iloc[:, 0]
    video_ids = csv.iloc[:, 1]
    start_times = csv.iloc[:, 3]
    end_times = csv.iloc[:, 4]
    json_dict = []
    for i in range(len(label)):
        vid = video_ids[i]
        start_time = start_times[i]
        end_time = end_times[i]
        for j in range(0, start_time):
            json_dict.append({
                'video_id': vid,
                'frame_id': j,
                'audio_name': f'audio_{j}.wav',
                'frame_name': f'frame_{j}.jpg',
                'answer': 'Background',
                'label': gt_labels.index('Background')
            })
        for j in range(start_time, end_time):
            json_dict.append({
                'video_id': vid,
                'audio_id': f'audio_{j}.wav',
                'frame_id': f'frame_{j}.jpg',
                'answer': label[i],
                'label': gt_labels.index(label[i])
            })
        for j in range(end_time, 10):
            json_dict.append({
                'video_id': vid,
                'audio_id': f'audio_{j}.wav',
                'frame_id': f'frame_{j}.jpg',
                'answer': 'Background',
                'label': gt_labels.index('Background')
            })
    json.dump(json_dict, open(json_path, 'w', encoding='utf8'), indent=4)


if __name__ == '__main__':
    data_dir = 'D:/Research/AVE/AVE_Dataset/'
    for split in ['train', 'val', 'test']:
        csv_to_json(f'{data_dir}/{split}Set.txt', f'{data_dir}/{split}.json')
