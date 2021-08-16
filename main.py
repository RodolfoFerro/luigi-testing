from google_drive_downloader import GoogleDriveDownloader as gdd
import numpy as np
import luigi
import cv2

import datetime
import secrets
import json


class GenerateInitialConfig(luigi.Task):

    def output(self):
        return luigi.LocalTarget('config.json')

    def run(self):
        initial_config = {
            '_id': secrets.token_hex(8),
            'video_name': 'video_7.mp4',
            'timestamps': {
                'database': str(datetime.datetime.now())
            },
            'product': np.random.choice(['grape', 'tomato', 'eggplant']),
        }

        with self.output().open('w') as f:
                json.dump(initial_config, f, ensure_ascii=False, indent=4)


class VideoDownload(luigi.Task):

    def requires(self):
        return GenerateInitialConfig()

    def output(self):
        return luigi.LocalTarget('config.json')

    def run(self):
        with self.output().open('r') as f:
            json_data = json.load(f)
        
        json_data['timestamps']['video'] = str(datetime.datetime.now())
        gdd.download_file_from_google_drive(
            file_id='XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX',
            dest_path='./' + json_data['video_name']
        )

        vid = cv2.VideoCapture(json_data['video_name'])
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid.release()

        json_data['flip'] = True if width > height else False

        with self.output().open('w') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)


class ModelDetails(luigi.Task):

    def requires(self):
        return VideoDownload()

    def output(self):
        return luigi.LocalTarget('config.json')

    def run(self):
        with self.output().open('r') as f:
            json_data = json.load(f)
        
        json_data['timestamps']['model'] = str(datetime.datetime.now())
        json_data['model'] = {
            'weights': './checkpoints/yolov4-{}_v4'.format(json_data['product']),
            'version': 'yolov4',
            'size': 416,
            'iou_threshold': 0.45,
            'score_threshold': 0.45,
            'tiny': False
        }

        with self.output().open('w') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)


class DistributionSampler(luigi.Task):

    def requires(self):
        return ModelDetails()

    def output(self):
        return luigi.LocalTarget('config.json')

    def run(self):
        with self.output().open('r') as f:
            json_data = json.load(f)
        
        json_data['timestamps']['sampler'] = str(datetime.datetime.now())
        json_data['sampler'] = {
            'batch_size': 20,
            'dist_name': 'gamma',
            'eps': 1e-2,
            'lower_percentile': 0.20,
            'upper_percentile': 0.95
        }

        with self.output().open('w') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)