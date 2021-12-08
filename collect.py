import cv2
import wget
import json
import os
from logging import error
import youtube_dl as ydl
from youtube_dl.utils import DownloadError


def _extract_video_id(url=''):
    index = url.index('?v=') + 3
    return url[index:]


_MSASL_VIDEOS_DIR = 'data/asl/youtube'


def _downloaded_video_ids():
    videos = os.listdir(_MSASL_VIDEOS_DIR)
    return set([video[:-4] for video in videos])


def _video_urls():
    datasetType = ["train", "val", "test"]
    urls = set()
    for type in datasetType:
        f = open(f"data/asl/MSASL_{type}.json", encoding="utf-8")
        data = json.load(f)
        urls = urls.union({it['url'] for it in data})
        urls = set(urls)
        return {url for url in urls if _extract_video_id(url) not in _downloaded_video_ids()}


## Run this one!
def runASL():
    for video in _video_urls():
        collectASLData(video)


def collectASLData(data):
    ydl_opts = {'outtmpl': 'data/asl/youtube/%(id)s.%(ext)s'}
    with ydl.YoutubeDL(ydl_opts) as video:
        try:
            video.download([data])
        except DownloadError:
            error(f'Video at {data} could not be downloaded.')


runASL()


## Run this one!
def runVSL():
    f = open("data/vsl/dictionary.json", encoding="utf-8")
    data = json.load(f)
    collectVSLData(data)


def collectVSLData(data):
    for i in data["data"]:
        video = wget.download(f"https://qipedc.moet.gov.vn/videos/{i['_id']}.mp4", out="data/vsl/datasets")
        video = video.split("/")[-1]
        cap = cv2.VideoCapture(f"data/vsl/datasets/{video}")
        cnt = 0

        w_frame, h_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)

        x, y, h, w = 250, 50, 650, 750

        # output
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(f"data/vsl/cropped/{video}", fourcc, fps, (w, h))

        while (cap.isOpened()):
            ret, frame = cap.read()

            cnt += 1

            if ret == True:
                crop_frame = frame[y:y + h, x:x + w]

                xx = cnt * 100 / frames
                # print(int(xx), '%')

                out.write(crop_frame)

                # Just to see the video in real time
                # cv2.imshow('croped', crop_frame)
                #
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
            else:
                break

        # cap.release()
        out.release()
        # cv2.destroyAllWindows()
