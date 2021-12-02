import cv2
import wget
import json

f = open("data/dictionary.json", encoding="utf-8")

data = json.load(f)




def collectData():
    for i in data["data"]:
        video = wget.download(f"https://qipedc.moet.gov.vn/videos/{i['_id']}.mp4", out="data/datasets")
        video = video.split("/")[-1]
        cap = cv2.VideoCapture(f"data/datasets/{video}")
        cnt = 0

        w_frame, h_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)

        x, y, h, w = 250, 50, 650, 750

        # output
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(f"data/cropped/{video}", fourcc, fps, (w, h))

        while (cap.isOpened()):
            ret, frame = cap.read()

            cnt += 1

            if ret == True:
                crop_frame = frame[y:y + h, x:x + w]

                xx = cnt * 100 / frames
                print(int(xx), '%')

                out.write(crop_frame)

                # Just to see the video in real time
                cv2.imshow('croped', crop_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()


collectData()
